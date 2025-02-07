import os
import json
from typing import (
    Optional,
    Any,
    Tuple,
    Iterable,
)
import copy
import logging
import optuna
import numpy as np
from collections import defaultdict
from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
import threading
import traceback
from cognify.optimizer.utils import _cognify_tqdm as tqdm
from cognify.optimizer.utils import _report_cost_reduction, _report_quality_impv, _report_latency_reduction
import heapq

from cognify._signal import _should_exit
from cognify.graph.program import Module
from cognify.hub.cogs.common import (
    CogBase,
    DynamicCogBase,
    EvolveType,
    AddNewModuleImportInterface,
)
from cognify.hub.cogs.utils import dump_params, load_params
from cognify.optimizer.evaluator import (
    EvaluationResult,
    EvaluatorPlugin,
    EvalTask,
    GeneralEvaluatorInterface,
)
from optuna.samplers import TPESampler, _base
from optuna.trial import TrialState, FrozenTrial
from cognify.optimizer.bo.tpe import FrugalTPESampler
from cognify.optimizer.rl.rl_sampler import RLSampler
from cognify.optimizer.core.flow import (
    TrialLog,
    ModuleTransformTrace,
    TopDownInformation,
    OptConfig,
)
from cognify.optimizer.control_param import SelectedObjectives
from cognify.optimizer.progress_info import pbar
from termcolor import colored

logger = logging.getLogger(__name__)


qc_identifier = "_#cognify_quality_constraint"


def get_quality_constraint(trial: optuna.trial.FrozenTrial):
    return trial.user_attrs.get(qc_identifier, (1,))


class OptimizationLayer:
    trial_log_cls = TrialLog
    opt_logs: dict[str, TrialLog]

    def __init__(
        self,
        name: str,
        evaluator: GeneralEvaluatorInterface,
        objectives: SelectedObjectives,
        dedicate_params: list[CogBase] = [],
        universal_params: list[CogBase] = [],
        target_modules: Iterable[str] = None,
        save_ckpt_interval: int = 0,
        quality_constraint: Optional[float] = None,
        base_quality: Optional[float] = None,
        base_cost: Optional[float] = None,
        base_exec_time: Optional[float] = None,
        hierarchy_level: int = 0,
    ):
        """
        The optimization will always try to minimize the price and maximize the score
        Please make sure the evaluator is consistent with this.

        Args:
            name: name of the optimization layer

            evaluator: the evaluator that will be used to evaluate the proposal

            dedicate_params: a list of params that is dedicated to pre-defined modules
                need to set `module_name` correctly

            universal_params: a list of params that will be broadcasted to all modules
                will ignore `module_name` field

            target_modules: if provided, only the modules in this list will be optimized
                this has higher priority than dedicated params

            save_ckpt_interval: if > 0, will save the optimization state every interval
                currently will overwrite the same file
        """
        self.name = name
        self.evaluator = evaluator
        self.objectives = objectives
        self.dedicate_params = dedicate_params
        self.universal_params = universal_params
        if len(self.dedicate_params) + len(self.universal_params) == 0:
            raise ValueError("No params provided for optimization")

        self.opt_direction = "maximize"
        self.target_modules = (
            set(target_modules) if target_modules is not None else None
        )
        self.opt_cost = 0

        # will be updated when prepare_opt_env is called
        self.params: dict[str, list[CogBase]] = None
        self.param_categorical_dist: dict[
            str, optuna.distributions.CategoricalDistribution
        ] = None

        self.opt_logs: dict[str, TrialLog] = dict()
        self.study: optuna.study.Study = None
        self._study_lock: threading.Lock = None
        self.opt_target_lm_names: set[str] = None
        self.save_ckpt_interval = save_ckpt_interval
        self.top_down_info: TopDownInformation = None

        self._best_score = None
        self._lowest_cost = None
        self._lowest_exec_time = None
        self.quality_constraint = quality_constraint
        self.base_quality = base_quality
        self.base_cost = base_cost
        self.base_exec_time = base_exec_time
        self.hierarchy_level = hierarchy_level
        self._should_stop = False # flag to early stop when convergence
        self._patience_budget = None # number of iterations to wait for improvement before early stop

    def prepare_opt_env(self):
        self.params = defaultdict(list)

        # NOTE: if param file exists, will load params from file and ignore the given params
        param_save_path = self.top_down_info.opt_config.param_save_path
        if os.path.exists(param_save_path):
            logger.info(f"Loading {self.name} params from {param_save_path}")
            l_param = load_params(param_save_path)
            for param in l_param:
                self.params[param.module_name].append(param)
            allowed_lm_names = set(param.module_name for param in l_param)
        else:
            module_pool = self.top_down_info.current_module_pool
            # extract mappings from old_lm_name to all_new_lm_names
            old_2_new_lms: dict[str, list[str]] = defaultdict(list)
            allowed_lm_names = set()
            for new_module in module_pool.values():
                old_name, new_modules = (
                    self.top_down_info.module_ttrace.get_derivatives_of_same_type(
                        new_module
                    )
                )
                new_names = [x.name for x in new_modules]
                if self.target_modules and old_name not in self.target_modules:
                    continue
                old_2_new_lms[old_name].extend(new_names)
                allowed_lm_names.update(new_names)

            # broadcast universal params
            if self.universal_params:
                for lm_name in allowed_lm_names:
                    params_cpy = copy.deepcopy(self.universal_params)
                    for param in params_cpy:
                        param.module_name = lm_name
                    self.params[lm_name] = params_cpy

            # apply dedicated params
            if self.dedicate_params:
                for param in self.dedicate_params:
                    target_names = []
                    if param.module_name in old_2_new_lms:
                        target_names = old_2_new_lms[param.module_name]
                    elif param.module_name in allowed_lm_names:
                        target_names = [param.module_name]

                    for lm_name in target_names:
                        mapped_param = copy.deepcopy(param)
                        mapped_param.module_name = lm_name
                        self.params[lm_name].append(mapped_param)

        self.param_categorical_dist = {
            param.hash: optuna.distributions.CategoricalDistribution(
                list(param.options.keys())
            )
            for _, params in self.params.items()
            for param in params
        }
        self.opt_target_lm_names = allowed_lm_names
        self.study = self.init_study()
        self._study_lock = threading.Lock()

    def param_cost_estimator(self, trial_proposal: dict[str, Any]) -> float:
        """get the cost of the trial proposal

        NOTE: trial proposal may not contain all params, e.g. if param only have single option or is sampled independently
        """
        total_cost = 0.0
        # convert to external params
        ext_trial_proposal = {}
        for param_name, dist in self.param_categorical_dist.items():
            if dist.single():
                continue
            if isinstance(trial_proposal[param_name], str):
                internal_repr = dist.to_internal_repr(
                    trial_proposal[param_name]
                )
            else:
                internal_repr = trial_proposal[param_name]

            ext_trial_proposal[param_name] = dist.to_external_repr(
                internal_repr
            )
                
        for lm_name, params in self.params.items():
            agent_cost = 1.0
            # for param imposed on the same agent, multiply the cost
            for param in params:
                if param.hash not in ext_trial_proposal:
                    continue
                selected = ext_trial_proposal[param.hash]
                option = param.options.get(selected, None)
                if option:
                    agent_cost *= option.cost_indicator
            total_cost += agent_cost
        return total_cost

    def init_study(
        self,
        old_study: optuna.Study = None,
        old_trials: list[optuna.trial.FrozenTrial] = None,
    ):
        """Create a new study and migrate old trials if provided

        For all provided trials, the params dist will be adjusted to the current
        self.params when this method is called.

        Recommand using name based options instead of index based options as the dynamic
        params update may change the mapping between option index and the option itself
        """
        qc_fn = get_quality_constraint if self.quality_constraint is not None else None
        assert self.top_down_info.opt_config.rl_based
        if self.top_down_info.opt_config.rl_based:
            sampler = RLSampler(
                cost_estimator=self.param_cost_estimator, 
                epsilon=0.1, 
                alpha=0.1,
                n_startup_trials=5,
                constraints_func=qc_fn
            )
        elif self.top_down_info.opt_config.frugal_eval_cost:
            sampler = FrugalTPESampler(
                cost_estimator=self.param_cost_estimator,
                multivariate=True,
                n_startup_trials=5,
                constraints_func=qc_fn,
            )
        else:
            sampler = TPESampler(
                multivariate=True, n_startup_trials=5, constraints_func=qc_fn
            )

        new_study = optuna.create_study(
            directions=self.objectives.get_optimization_directions(),
            sampler=sampler,
        )

        f_trials: list[optuna.trial.FrozenTrial] = []
        if old_study:
            f_trials.extend(old_study.trials)
        if old_trials:
            f_trials.extend(old_trials)

        for trial in f_trials:
            # Modify previous trials.
            # user and system attr will be copied
            # no need to deal with them here
            dists = self.param_categorical_dist
            trial.distributions = dists
            # Persist the changes to the storage (in a new study).
            new_study.add_trial(trial)
        return new_study

    def _apply_params(
        self,
        trial_params: dict[str, Any],
        program_copy: list[Module],
    ) -> tuple[list[Module], ModuleTransformTrace]:
        trace_for_next_level = copy.deepcopy(self.top_down_info.module_ttrace)
        opt_target_lms = Module.all_with_predicate(
            program_copy, lambda m: m.name in self.opt_target_lm_names
        )
        module_dict = {lm.name: lm for lm in opt_target_lms}
        new_modules = []
        changed_modules = set()
        for lm_name, params in self.params.items():
            for param in params:
                selected = trial_params[param.hash]
                try:
                    new_module, new_mapping = param.apply_option(
                        selected, module_dict[lm_name]
                    )
                    for old_name, new_name in new_mapping.items():
                        trace_for_next_level.add_mapping(old_name, new_name)
                    new_modules.append(new_module)
                    changed_modules.add(lm_name)
                    trace_for_next_level.register_proposal(
                        self.name, [(lm_name, param.name, selected)]
                    )
                except Exception as e:
                    logger.error(
                        f"Error in applying param {param.name} to {lm_name}: {e}"
                    )
                    logger.error(
                        f"Module dict {module_dict.keys()} and self params {self.params.keys()}"
                    )
                    raise


        for m_name, new_module in module_dict.items():
            if m_name not in changed_modules:
                new_modules.append(new_module)
        return new_modules, trace_for_next_level

    def generate_trial_id(self) -> tuple[str, int]:
        # always increment the trial number
        # optuna trial id start from num_trials, if previous run is interrupted
        # using optuna trial number will have conflict
        with self._study_lock:
            max_id = None
            for log in self.opt_logs.values():
                id = int(log.id.split("_")[-1])
                max_id = id if max_id is None else max(max_id, id)
        new_trial_number = max_id + 1 if max_id is not None else 0
        return ".".join(
            self.top_down_info.trace_back + [f"{self.name}_{new_trial_number}"]
        )

    def propose(
        self,
        ms: list[Module],
        n_sample: int,
    ) -> list[Tuple[optuna.trial.Trial, list[Module], ModuleTransformTrace, str]]:
        """Propse and apply the next set of params

        Will return new set of modules without modifying the given modules
        """
        next_to_run = []
        for i in range(n_sample):
            with self._study_lock:
                trial = self.study.ask(self.param_categorical_dist)

            logger.debug(
                f"- {self.name} - apply param - Trial {trial.number} params: {trial.params}"
            )
            trial_id_str = self.generate_trial_id()
            trial_log = self.trial_log_cls(
                params=trial.params, bo_trial_id=trial.number, id=trial_id_str
            )

            self.opt_logs[trial_log.id] = trial_log
            program_copy = copy.deepcopy(ms)
            new_modules, new_trace = self._apply_params(trial.params, program_copy)
            next_to_run.append((trial, new_modules, new_trace, trial_log.id))
        return next_to_run

    def evaluate(
        self,
        log_id: str,
        new_top_down_info: TopDownInformation,
    ) -> EvaluationResult:
        eval_task = EvalTask.from_top_down_info(new_top_down_info)
        eval_result: EvaluationResult = self.evaluator.evaluate(eval_task)
        return eval_result

    def add_constraint(self, score, trial: optuna.trial.Trial):
        # Soft constraint, if score is lower than the quality constraint, reject it
        if self.quality_constraint is not None:
            constraint_result = (self.quality_constraint - score,)
            trial.set_user_attr(qc_identifier, constraint_result)
            # NOTE: add system attr at loading time
            # trial.set_system_attr(_base._CONSTRAINTS_KEY, constraint_result)

    def _update_best_trial(self, eval_result: EvaluationResult):
        with self._study_lock:
            current_score, current_cost, current_exec_time = self.get_eval_feedback(eval_result)
            if current_score is not None and current_cost is not None and current_exec_time is not None:
                self._best_score = (
                    current_score if self._best_score is None else max(self._best_score, current_score)
                )
                self._lowest_cost = (
                    current_cost if self._lowest_cost is None else min(self._lowest_cost, current_cost)
                )
                self._lowest_exec_time = (
                    current_exec_time if self._lowest_exec_time is None else min(self._lowest_exec_time, current_exec_time)
                )

            if not self._should_stop and self.top_down_info.opt_config.patience is not None and self._patience_budget is not None:
                impv = False
                score_threshold = self.top_down_info.opt_config.patience.quality_min_delta
                cost_threshold = self.top_down_info.opt_config.patience.cost_min_delta
                # reset if score or cost is improved
                if current_score is not None and current_score >= self._best_score * (1 + score_threshold):
                    self._patience_budget = self.top_down_info.opt_config.patience.n_iterations
                    impv = True
                if current_cost is not None and current_cost <= self._lowest_cost * (1 - cost_threshold):
                    self._patience_budget = self.top_down_info.opt_config.patience.n_iterations
                    impv = True
                if not impv:
                    self._patience_budget -= 1
                    # early stop if patience budget is used up
                    if self._patience_budget <= 0:
                        self._should_stop = True

    def update(
        self,
        trial: optuna.trial.Trial,
        eval_result: EvaluationResult,
        log_id: str,
    ):
        # if eval is interrupted, the result will not be used
        if not eval_result.complete:
            return

        score, price, exec_time = eval_result.reduced_score, eval_result.reduced_price, eval_result.reduced_exec_time
        self.opt_logs[log_id].score = score
        self.opt_logs[log_id].price = price
        self.opt_logs[log_id].exec_time = exec_time

        self.opt_logs[log_id].eval_cost = eval_result.total_eval_cost
        logger.debug(
            f"- {self.name} - Trial {trial.number} result: score= {score:.2f}, cost@1000= ${price*1000:.3f}, exec_time= {exec_time:.2f}s"
        )
        self.opt_cost += eval_result.total_eval_cost

        self._update_best_trial(eval_result)

        # update study if any dynamic params can evolve
        with self._study_lock:
            self.add_constraint(score, trial)
            frozen_trial = self.study.tell(trial, self.objectives.select_from(score, price, exec_time))
            is_evolved = False
            for params in self.params.values():
                for param in params:
                    if isinstance(param, DynamicCogBase):
                        evolve_type = param.evolve(eval_result)
                        if evolve_type != EvolveType.ID:
                            is_evolved = True
            if is_evolved:
                # update param dist
                self.param_categorical_dist = {
                    param.hash: optuna.distributions.CategoricalDistribution(
                        list(param.options.keys())
                    )
                    for _, params in self.params.items()
                    for param in params
                }
                # create new study and migrate all trials
                new_study = self.init_study(self.study)
                self.study = new_study
        self.opt_logs[log_id].finished = True

    def get_eval_feedback(self, eval_result: EvaluationResult):
        """Override this method to get feedback from the evaluation result

        Bottom Layer uses the average score and price on the evaluation set
        Upper Layer uses the best score and price on the inner loop optimization
        """
        raise NotImplementedError

    def _get_new_python_paths(self):
        new_python_paths = []
        for lm_name, params in self.params.items():
            for param in params:
                if isinstance(param, AddNewModuleImportInterface):
                    new_python_paths.extend(param.get_python_paths())
        return new_python_paths

    def save_ckpt(self, opt_log_path: str, param_save_path: str):
        opt_logs_json_obj = {}
        for k, v in self.opt_logs.items():
            if v.finished:
                opt_logs_json_obj[k] = v.to_dict()
        if not opt_logs_json_obj:
            logger.warning("No finished trials to save")
            return
        json.dump(opt_logs_json_obj, open(opt_log_path, "w+"), indent=4)
        params = [param for params in self.params.values() for param in params]
        dump_params(params, param_save_path)

    def prepare_next_level_tdi(
        self,
        new_program: list[Module],
        new_trace: ModuleTransformTrace,
        trial_id: str,
    ) -> TopDownInformation:
        """create info for next level optimization or actual evaluation

        NOTE: default implementation does not set opt_config for next level
        bottom layer works fine but outer layer needs to reset themselves
        """

        # add new python paths incase new module imports are added
        python_paths = (
            self.top_down_info.other_python_paths + self._get_new_python_paths()
        )
        python_paths = list(set(python_paths))

        next_level_info = TopDownInformation(
            opt_config=copy.deepcopy(self.top_down_info.opt_config),
            all_params=self.top_down_info.all_params.copy(),  # params from upper-levels will not be changed
            module_ttrace=new_trace,
            current_module_pool={m.name: m for m in new_program},
            script_path=self.top_down_info.script_path,
            script_args=self.top_down_info.script_args,
            other_python_paths=python_paths,
        )
        next_level_info.trace_back.append(trial_id)

        # add current level params for next level
        for lm_name, params in self.params.items():
            # NOTE: params might be updated when scheduling the current iteration
            # so we make a copy of the current params
            for param in params:
                next_level_info.all_params[param.hash] = copy.deepcopy(param)

        return next_level_info

    def _optimize_iteration(
        self,
        base_program: list[Module],
        frac: float,
    ) -> EvaluationResult:
        next_trial, program, new_trace, log_id = self.propose(base_program, 1)[0]
        next_level_info = self.prepare_next_level_tdi(program, new_trace, log_id)
        next_level_info.opt_config.frac = frac

        if _should_exit():
            return None

        try:
            eval_result = self.evaluate(log_id, next_level_info)
            self.update(next_trial, eval_result, log_id)
            return eval_result
        except Exception as e:
            logger.error(f"Error in opt iteration: {e}")
            logger.error(traceback.format_exc())
            raise

    def _optimize(self, base_program: list[Module], frac: float):
        opt_config = self.top_down_info.opt_config
        frac = opt_config.frac
        num_current_trials = len(self.opt_logs)

        initial_score = self._best_score if self._best_score is not None else 0.0
        initial_cost = self._lowest_cost if self._lowest_cost is not None else 0.0
        initial_exec_time = self._lowest_exec_time if self._lowest_exec_time is not None else 0.0

        if self.hierarchy_level == 0:
            pbar.init_pbar(
                total=float(num_current_trials + opt_config.n_trials),
                initial=float(num_current_trials),
                initial_score=initial_score,
                initial_cost=initial_cost,
                initial_exec_time=initial_exec_time,
                opt_cost=self.opt_cost
            )

        counter = 0
        if opt_config.throughput == 1:
            for _ in range(opt_config.n_trials):
                if self._should_stop:
                    pbar.finish()
                if _should_exit() or self._should_stop:
                    break
                result = self._optimize_iteration(base_program, frac / opt_config.n_trials)

                if result is None or not result.complete:
                    continue
                counter += 1
                if (
                    self.save_ckpt_interval > 0
                    and counter % self.save_ckpt_interval == 0
                ):
                    self.save_ckpt(
                        opt_config.opt_log_path, opt_config.param_save_path
                    )
                pbar.update_status(self._best_score, self._lowest_cost, self._lowest_exec_time, self.opt_cost)
        else:
            with ThreadPoolExecutor(max_workers=opt_config.throughput) as executor:
                futures = [
                    executor.submit(self._optimize_iteration, base_program, frac / opt_config.n_trials)
                    for _ in range(opt_config.n_trials)
                ]
                for f in as_completed(futures):
                    try:
                        result = f.result()
                        if result and result.complete:
                            counter += 1
                            if (
                                self.save_ckpt_interval > 0
                                and counter % self.save_ckpt_interval == 0
                            ):
                                self.save_ckpt(
                                    opt_config.opt_log_path,
                                    opt_config.param_save_path,
                                )
                            pbar.update_status(self._best_score, self._lowest_cost, self._lowest_exec_time, self.opt_cost)
                        if self._should_stop:
                            pbar.finish()
                        if _should_exit() or self._should_stop:
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                    except Exception as e:
                        logger.error(f"Error in evaluating task: {e}", exc_info=True)
                        raise
        pbar.release_position(self.hierarchy_level)

    def get_finished_bo_trials(self, need_copy: bool) -> list[FrozenTrial]:
        states_of_interest = (TrialState.COMPLETE,)
        return self.study.get_trials(deepcopy=need_copy, states=states_of_interest)

    def get_pareto_front(self, candidates: list[TrialLog, str]) -> list[tuple[TrialLog, str]]:
        """Find the pareto-efficient points

        Each with their config log path. This is for upper level to show correct bottom level config since the path is generated randomly for each innerloop.

        This function will not filter with constraints
        """
        if not candidates:
            return []
        score_cost_time_list = []
        for trial_log, log_path in candidates:
            score_cost_time_list.append((trial_log.score, trial_log.price, trial_log.exec_time))

        vectors = np.array([self.objectives.select_from(-score, price, exec_time) for score, price, exec_time in score_cost_time_list])
        is_efficient = np.ones(vectors.shape[0], dtype=bool)
        for i, v in enumerate(vectors):
            if is_efficient[i]:
                is_efficient[is_efficient] = np.any(
                    vectors[is_efficient] < v, axis=1
                )  # Keep any point with a lower cost
                is_efficient[i] = True  # And keep self

        # return filtered [T_ParetoProgram]
        pareto_frontier = [
            (log, path) for (log, path), eff in zip(candidates, is_efficient) if eff
        ]
        return pareto_frontier

    def pre_optimize(self): ...

    def get_all_candidates(self):
        cancidates = []
        for log_id, log in self.opt_logs.items():
            if not log.finished:
                continue
            # if not meet the quality constraint, skip
            if (
                self.quality_constraint is not None
                and log.score < self.quality_constraint
            ):
                continue
            cancidates.append((log, self.top_down_info.opt_config.opt_log_path))
        return cancidates

    def post_optimize(self):
        # Analysis optimization result
        candidates = self.get_all_candidates()
        pareto_frontier = self.get_pareto_front(candidates=candidates)
        if self.hierarchy_level == 0:
            print(f"================ Optimization Results =================")
            print(f"Optimized for: {str(self.objectives)}")

            if len(pareto_frontier) == 0:
                print("Based on current optimization parameters, the best solution is the original workflow.")
                print("We recommend increasing the number of trials or relaxing constraints.")
            else:
                print(f"Number of Optimized Workflows Generated: {len(pareto_frontier)}")

            for i, (trial_log, log_path) in enumerate(pareto_frontier):
                print("--------------------------------------------------------")
                print("Optimization_{}".format(i + 1))
                # logger.info("  Params: {}".format(trial_log.params))
                if self.base_quality is not None:
                    print("  Quality improvement: {:.0f}%".format(_report_quality_impv(trial_log.score, self.base_quality)))
                if self.base_cost is not None:
                    print("  Cost: {:.2f}x original".format(_report_cost_reduction(trial_log.price, self.base_cost)))
                if self.base_exec_time is not None:
                    print("  Execution time: {:.2f}x original".format(_report_latency_reduction(trial_log.exec_time, self.base_exec_time)))
                print("  Quality: {:.2f}, Cost per 1K invocation: ${:.2f}, Execution time: {:.2f}s".format(trial_log.score, trial_log.price * 1000, trial_log.exec_time))
                    # print("  Applied at: {}".format(trial_log.id))
                    # logger.info("  config saved at: {}".format(log_path))

                # logger.info("Opt Cost: {}".format(self.opt_cost))
                print("========================================================")
        return pareto_frontier

    def _load_opt_ckpt(self, opt_log_path: str):
        self.load_opt_log(opt_log_path)

        if self.opt_logs:
            candidates = self.get_all_candidates()
            if candidates:
                self._best_score = max([log.score for log, _ in candidates])
                self._lowest_cost = min([log.price for log, _ in candidates])
                self._lowest_exec_time = min([log.exec_time for log, _ in candidates])

        for trial_log_id, trial_log in self.opt_logs.items():
            assert trial_log.finished, f"Trial {trial_log_id} is not finished"
            trial = optuna.trial.create_trial(
                params=trial_log.params,
                values=self.objectives.select_from(trial_log.score, trial_log.price, trial_log.exec_time),
                distributions=self.param_categorical_dist,
            )
            if self.quality_constraint is not None:
                self.add_constraint(trial_log.score, trial)
                trial.set_system_attr(
                    _base._CONSTRAINTS_KEY, get_quality_constraint(trial)
                )
            self.study.add_trial(trial)

    def load_opt_log(self, opt_log_path: str):
        with open(opt_log_path, "r") as f:
            opt_trace = json.load(f)
        for trial_log_id, trial_meta in opt_trace.items():
            trial_log = self.trial_log_cls.from_dict(trial_meta)
            self.opt_logs[trial_log_id] = trial_log
            self.opt_cost += trial_log.eval_cost

    def optimize(
        self,
        current_tdi: TopDownInformation,
    ) -> tuple[float, list[TrialLog], dict[str, TrialLog]]:
        self.opt_cost = 0

        # prepare optimization environment
        current_tdi.initialize()
        self.top_down_info = current_tdi
        if current_tdi.opt_config.patience is not None:
            self._patience_budget = current_tdi.opt_config.patience.n_iterations
        self.prepare_opt_env()

        # load previous optimization logs if exists
        opt_log_path = self.top_down_info.opt_config.opt_log_path
        if os.path.exists(opt_log_path):
            self._load_opt_ckpt(opt_log_path)

        # start optimization
        total_budget = self.top_down_info.opt_config.n_trials
        if total_budget > 0:
            self.pre_optimize()
            logger.debug(f"Start optimization {self.name} with {total_budget} trials")
            self._optimize(list(current_tdi.current_module_pool.values()), current_tdi.opt_config.frac)
            logger.debug(f"Optimization {self.name} finished")
            self.save_ckpt(
                self.top_down_info.opt_config.opt_log_path,
                self.top_down_info.opt_config.param_save_path,
            )

        pareto_frontier = self.post_optimize()
        finished_opt_logs = {k: v for k, v in self.opt_logs.items() if v.finished}
        return self.opt_cost, pareto_frontier, finished_opt_logs

    def easy_optimize(
        self,
        opt_config: OptConfig,
        script_path: str,
        script_args: Optional[list[str]] = None,
        other_python_paths: Optional[list[str]] = None,
    ):
        tdi = TopDownInformation(
            opt_config=opt_config,
            all_params=None,
            module_ttrace=None,
            current_module_pool=None,
            script_path=script_path,
            script_args=script_args,
            other_python_paths=other_python_paths,
        )
        return self.optimize(tdi)


class BottomLevelTrialLog(TrialLog):
    def __init__(
        self,
        params,
        bo_trial_id,
        id=None,
        score=0,
        price=0,
        exec_time=0,
        eval_cost=0,
        finished=False,
        eval_task: dict = None,
    ):
        super().__init__(params, bo_trial_id, id, score, price, exec_time, eval_cost, finished)
        self.eval_task = eval_task

    def to_dict(self):
        return {
            **super().to_dict(),
            "eval_task": self.eval_task,
        }

    def show_transformation(self) -> str:
        eval_task = EvalTask.from_dict(self.eval_task)
        return eval_task.show_opt_trace()

    def compare_using_objective(self, other: "BottomLevelTrialLog", objectives: SelectedObjectives):
        if objectives.quality:
            return self.score > other.score
        if objectives.cost:
            return self.price < other.price
        if objectives.latency:
            return self.exec_time < other.exec_time


class BottomLevelOptimization(OptimizationLayer):
    opt_logs: dict[str, BottomLevelTrialLog]
    evaluator: EvaluatorPlugin
    trial_log_cls = BottomLevelTrialLog

    def evaluate(self, log_id, new_top_down_info):
        eval_task = EvalTask.from_top_down_info(new_top_down_info)
        self.opt_logs[log_id].eval_task = eval_task.to_dict()
        eval_result: EvaluationResult = self.evaluator.evaluate(eval_task, frac=new_top_down_info.opt_config.frac)
        return eval_result

    def get_best_trial_log(self) -> BottomLevelTrialLog:
        best_trial_log: BottomLevelTrialLog = None
        for log in self.opt_logs.values():
            if best_trial_log is None or log.compare_using_objective(best_trial_log, self.objectives):
                best_trial_log = log
        return best_trial_log

    def update(
        self,
        trial: optuna.trial.Trial,
        eval_result: EvaluationResult,
        log_id: str,
    ):
        if not eval_result.complete:
            return
        score, price, exec_time = eval_result.reduced_score, eval_result.reduced_price, eval_result.reduced_exec_time
        self.opt_logs[log_id].score = score
        self.opt_logs[log_id].price = price
        self.opt_logs[log_id].exec_time = exec_time

        self.opt_logs[log_id].eval_cost = eval_result.total_eval_cost
        logger.debug(
            f"- {self.name} - Trial {trial.number} result: score= {score:.2f}, cost@1000= ${price*1000:.3f}, exec_time= {exec_time:.2f}s"
        )
        self.opt_cost += eval_result.total_eval_cost

        self._update_best_trial(eval_result)
        # update study if any dynamic params can evolve
        with self._study_lock:
            self.add_constraint(score, trial)
            frozen_trial = self.study.tell(trial, self.objectives.select_from(score, price, exec_time))
            existing_trials = self.get_finished_bo_trials(False)
            if (
                len(existing_trials) > 0
                and len(existing_trials) % self.top_down_info.opt_config.evolve_interval
                == 0
            ):
                logger.debug(
                    f"Evolving on Validation Set at {len(existing_trials)} trials"
                )
                evolve_result = eval_result
                if self.evaluator.dataset["eval"][0] is not None:
                    logger.debug("Use best score config to get evolving results")
                    # use best score config to get evolving results
                    best_trial_log = self.get_best_trial_log()
                    evolve_eval_task = EvalTask.from_dict(
                        copy.deepcopy(best_trial_log.eval_task)
                    )
                    evolve_result = self.evaluator.get_score(
                        mode="eval", task=evolve_eval_task,
                    )
                    logger.debug(
                        f"Validation set result: score= {evolve_result.reduced_score:.2f}, cost@1000= {evolve_result.reduced_price*1000:.3f}, exec_time= {evolve_result.reduced_exec_time:.2f}s"
                    )

                is_evolved = False
                for params in self.params.values():
                    for param in params:
                        if isinstance(param, DynamicCogBase):
                            evolve_type = param.evolve(evolve_result)
                            if evolve_type != EvolveType.ID:
                                is_evolved = True
                if is_evolved:
                    # update param dist
                    self.param_categorical_dist = {
                        param.hash: optuna.distributions.CategoricalDistribution(
                            list(param.options.keys())
                        )
                        for _, params in self.params.items()
                        for param in params
                    }
                    # create new study and migrate all trials
                    new_study = self.init_study(self.study)
                    self.study = new_study
        self.opt_logs[log_id].finished = True

    def get_eval_feedback(self, eval_result: EvaluationResult):
        avg_score = eval_result.reduced_score
        if self.quality_constraint is None or avg_score >= self.quality_constraint:
            return avg_score, eval_result.reduced_price, eval_result.reduced_exec_time
        return None, None, None

    def pre_optimize(self):
        """Bootstrap the initial evolving params with given config

        If already load trials from saved opt log, will skip this step
        """
        if len(self.get_finished_bo_trials(False)) > 0:
            return

        has_evolve = False
        for params in self.params.values():
            for param in params:
                if isinstance(param, DynamicCogBase):
                    has_evolve = True
                    break
        if not has_evolve:
            return

        logger.debug(f"Start pre-optimization for {self.name}")
        eval_task = EvalTask.from_top_down_info(self.top_down_info)
        eval_task.trace_back = ["pre-opt-analysis"]
        if self.evaluator.dataset["eval"][0] is None:
            eval_result = self.evaluator.get_score(
                mode="train",
                task=eval_task,
            )
        else:
            eval_result = self.evaluator.get_score(
                mode="eval",
                task=eval_task,
            )
        with self._study_lock:
            is_evolved = False
            for params in self.params.values():
                for param in params:
                    if isinstance(param, DynamicCogBase):
                        evolve_type = param.evolve(eval_result)
                        if evolve_type != EvolveType.ID:
                            is_evolved = True
            if is_evolved:
                # update param dist
                self.param_categorical_dist = {
                    param.hash: optuna.distributions.CategoricalDistribution(
                        list(param.options.keys())
                    )
                    for _, params in self.params.items()
                    for param in params
                }
                # create new study and migrate all trials
                new_study = self.init_study(self.study)
                self.study = new_study

    @staticmethod
    def easy_eval(
        evaluator: EvaluatorPlugin,
        trial_id: str,
        opt_log_path: str,
        base_quality: float = None,
        base_cost: float = None,
        base_exec_time: float = None
    ) -> EvaluationResult:
        if not os.path.exists(opt_log_path):
            raise ValueError(f"Opt log path {opt_log_path} does not exist")

        with open(opt_log_path, "r") as f:
            opt_trace = json.load(f)
        trial_log = BottomLevelTrialLog.from_dict(opt_trace[trial_id])

        # apply selected trial
        print(f"----- Testing select trial {trial_id} -----")
        print("  Params: {}".format(trial_log.params))
        # print("  Training Quality: {:.3f}, Cost per 1K invocation: ${:.2f}\n".format(trial_log.score, trial_log.price * 1000))

        eval_task = EvalTask.from_dict(trial_log.eval_task)
        # run evaluation
        eval_result = evaluator.get_score(mode='test', task=eval_task, show_progress_bar=True)

        print(f"=========== Evaluation Results ===========")
        if base_quality is not None:
            print("  Quality improvement: {:.0f}%".format(_report_quality_impv(eval_result.reduced_score, base_quality)))
        if base_cost is not None:
            print("  Cost is {:.2f}x original".format(_report_cost_reduction(eval_result.reduced_price, base_cost)))
        if base_exec_time is not None:
            print("  Execution time is {:.2f}x original".format(_report_latency_reduction(eval_result.reduced_exec_time, base_exec_time)))
        print("  Quality: {:.2f}, Cost per 1K invocation: ${:.2f}, Execution time: {:.2f}s".format(eval_result.reduced_score, eval_result.reduced_price * 1000, eval_result.reduced_exec_time))
        print("===========================================")

        return eval_result
