"""TravelPlanner Agent: GoalSeeking orchestrator with retrieval + plan assembler subagents.

Architecture:
- Top-level GoalSeeking agent orchestrates two phases:
  1. gather_information() -> delegates to RetrievalAgent (GoalSeeking)
  2. build_constrained_plan() -> delegates to PlanAssemblerAgent (DesignExecute)

The retrieval agent iteratively gathers flights, restaurants, accommodations,
attractions, distances from the reference database.

The plan assembler takes gathered data + constraints and deterministically
assembles a valid day-by-day plan using filtering, optimization, and cost
calculation primitives.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from opensymbolicai.blueprints import GoalSeeking
from opensymbolicai.core import decomposition, evaluator, primitive
from opensymbolicai.llm import LLM, LLMConfig
from opensymbolicai.models import (
    ExecutionResult,
    GoalContext,
    GoalEvaluation,
    GoalSeekingConfig,
)

from travelplanner_bench.plan_assembler import PlanAssemblerAgent
from travelplanner_bench.models import (
    GatheredData,
    TravelPlanContext,
    TravelPlannerTask,
)
from travelplanner_bench.retrieval_agent import RetrievalAgent
from travelplanner_bench.tools import ReferenceDatabase

log = logging.getLogger(__name__)


class TravelPlannerAgent(GoalSeeking):
    """Orchestrates travel planning through retrieval and plan assembly.

    Two-phase approach:
    Phase 1 - Information Gathering: Call gather_information() to search
      for flights, restaurants, accommodations, attractions, and distances.
      Returns a GatheredData object with all retrieved information.

    Phase 2 - Plan Assembly: Call build_constrained_plan() with the
      gathered data. The plan assembler deterministically filters,
      optimizes, and assembles a valid plan satisfying budget, room type,
      room rule, cuisine, and transportation constraints.

    RULES:
    1. Always call gather_information() first to collect all needed data.
    2. Then call build_constrained_plan() with the gathered data.
    3. The plan assembler handles ALL constraint satisfaction - do not
       try to build the plan manually.
    """

    def __init__(
        self,
        llm: LLMConfig | LLM,
        max_iterations: int = 5,
    ) -> None:
        super().__init__(
            llm=llm,
            name="TravelPlannerAgent",
            description=(
                "Orchestrates travel planning through two phases:\n\n"
                "Phase 1 - Call gather_information() to search for flights, "
                "restaurants, accommodations, attractions, and distances from "
                "the reference database.\n\n"
                "Phase 2 - Call build_constrained_plan() with the gathered data "
                "to deterministically assemble a valid plan that satisfies all "
                "constraints (budget, room type, room rule, cuisine, transportation).\n\n"
                "RULES:\n"
                "1. Always call gather_information() FIRST.\n"
                "2. Then call build_constrained_plan() with the result.\n"
                "3. Do NOT try to build plans manually - the plan assembler "
                "handles all constraint satisfaction deterministically."
            ),
            config=GoalSeekingConfig(max_iterations=max_iterations),
        )
        self._current_task: TravelPlannerTask | None = None
        self._db: ReferenceDatabase | None = None
        self._submitted_plan: list[dict[str, Any]] | None = None
        self._cached_gathered: GatheredData | None = None

    # =========================================================================
    # SUBAGENT PRIMITIVES
    # =========================================================================

    @primitive(read_only=True)
    def gather_information(self) -> GatheredData:
        """Gather all necessary travel data from the reference database.

        Delegates to the RetrievalAgent which iteratively searches for:
        - Flights (outbound and return)
        - Restaurants in each destination city
        - Accommodations in each destination city
        - Attractions in each destination city
        - Inter-city distances (for multi-city trips)
        - City lists (for state-level destinations)

        Returns cached data immediately if already gathered.

        Returns:
            GatheredData object with all retrieved information organized by city.
        """
        if self._cached_gathered is not None:
            return self._cached_gathered

        assert self._current_task is not None
        assert self._db is not None

        retrieval = RetrievalAgent(
            llm=self._llm_config if hasattr(self, "_llm_config") else self._llm,
            db=self._db,
        )
        gathered = retrieval.gather(self._current_task)
        self._cached_gathered = gathered
        return gathered

    @primitive(read_only=False)
    def build_constrained_plan(self, gathered_data: GatheredData) -> str:
        """Build a valid plan from gathered data, satisfying all constraints.

        Delegates to the PlanAssemblerAgent which deterministically:
        1. Filters accommodations by room type, room rule, minimum nights
        2. Filters restaurants by cuisine requirements
        3. Determines valid transport based on constraints
        4. Optimizes for cheapest valid options within budget
        5. Assembles a complete day-by-day plan

        Retries internally with error feedback on failure (up to 3 attempts).

        Args:
            gathered_data: GatheredData from gather_information().

        Returns:
            Confirmation string. The plan is submitted internally.
        """
        assert self._current_task is not None

        max_attempts = 3
        last_error: str | None = None

        for attempt in range(max_attempts):
            assembler = PlanAssemblerAgent(
                llm=self._llm_config if hasattr(self, "_llm_config") else self._llm,
            )
            plan = assembler.assemble_plan(
                gathered_data, self._current_task, previous_error=last_error
            )

            if plan:
                self._submitted_plan = plan
                return f"Plan built with {len(plan)} days."

            last_error = assembler.last_error or "set_plan() was never called"
            log.warning(
                "Plan assembly attempt %d/%d failed: %s",
                attempt + 1, max_attempts, last_error,
            )

        return f"Failed to build plan after {max_attempts} attempts: {last_error}"

    # =========================================================================
    # DECOMPOSITION EXAMPLES
    # =========================================================================

    @decomposition(
        intent="Plan a 3-day trip from Sarasota to Chicago for 1 person, budget $1900",
        expanded_intent=(
            "Two-phase approach: first gather all information, then build "
            "a constrained plan. Always gather first, then solve."
        ),
    )
    def _ex_orchestrate(self) -> str:
        gathered = self.gather_information()
        result = self.build_constrained_plan(gathered)
        return result

    # =========================================================================
    # GOALSEEKING OVERRIDES
    # =========================================================================

    def create_context(self, goal: str) -> TravelPlanContext:
        self._submitted_plan = None
        task = self._current_task
        return TravelPlanContext(
            goal=goal,
            query=task.query if task else goal,
            org=task.org if task else "",
            dest=task.dest if task else "",
            days=task.days if task else 3,
            people_number=task.people_number if task else 1,
            budget=task.budget if task else 0,
            local_constraint=task.local_constraint if task else {},
        )

    def update_context(
        self, context: GoalContext, execution_result: ExecutionResult
    ) -> None:
        assert isinstance(context, TravelPlanContext)

        for step in execution_result.trace.steps:
            if not step.success:
                if step.error:
                    context.solver_error = step.error
                continue

            prim = step.primitive_called

            if prim == "gather_information" and isinstance(
                step.result_value, GatheredData
            ):
                context.data_gathered = True
                context.gathered_summary = step.result_value.summary()

            elif prim == "build_constrained_plan":
                if self._submitted_plan is not None:
                    context.plan_built = True
                    context.current_plan = self._submitted_plan
                    context.plan_complete = True
                else:
                    # Plan assembly failed — capture the reason
                    result_str = str(step.result_value) if step.result_value else ""
                    context.solver_error = result_str or "Plan assembly failed"

    @evaluator
    def check_plan_ready(
        self, goal: str, context: GoalContext
    ) -> GoalEvaluation:
        assert isinstance(context, TravelPlanContext)
        return GoalEvaluation(goal_achieved=context.plan_complete)

    def _extract_final_answer(self, context: GoalContext) -> Any:
        assert isinstance(context, TravelPlanContext)
        return context.current_plan

    # =========================================================================
    # PUBLIC API
    # =========================================================================

    def solve(
        self, task: TravelPlannerTask
    ) -> tuple[list[dict[str, Any]] | None, int, list[dict[str, Any]]]:
        """Solve a single TravelPlanner task.

        Returns:
            Tuple of (plan_or_None, iteration_count, iteration_logs).
        """
        self._current_task = task
        self._db = ReferenceDatabase(task.reference_information)
        self._cached_gathered = None
        # Store LLM config for subagent creation
        self._llm_config = self._llm

        goal = self._build_goal_string(task)
        result = self.seek(goal)
        plan = result.final_answer

        # Build iteration logs for compatibility with runner
        iteration_logs: list[dict[str, Any]] = []
        for iteration in result.iterations:
            iter_log: dict[str, Any] = {
                "iteration": iteration.iteration_number,
                "prompt": "",
                "response": "",
                "extracted_code": "",
                "model": "",
                "input_tokens": 0,
                "output_tokens": 0,
                "time_seconds": 0.0,
                "steps": [],
                "goal_achieved": iteration.evaluation.goal_achieved,
            }
            gen = iteration.plan_result.plan_generation
            if gen:
                llm_int = gen.llm_interaction
                iter_log["prompt"] = llm_int.prompt
                iter_log["response"] = llm_int.response
                iter_log["extracted_code"] = gen.extracted_code
                iter_log["model"] = llm_int.model or iteration.plan_result.model
                iter_log["input_tokens"] = llm_int.input_tokens
                iter_log["output_tokens"] = llm_int.output_tokens
                iter_log["time_seconds"] = llm_int.time_seconds

            for step in iteration.execution_result.trace.steps:
                args_str = ", ".join(
                    f"{k}={v.resolved_value!r}"
                    for k, v in (step.args or {}).items()
                )
                step_log = {
                    "step": step.step_number,
                    "primitive": step.primitive_called or "?",
                    "args": args_str,
                    "result": repr(step.result_value)[:500],
                    "time_seconds": step.time_seconds,
                    "success": step.success,
                    "error": step.error,
                }
                iter_log["steps"].append(step_log)

            iteration_logs.append(iter_log)

        return plan, result.iteration_count, iteration_logs

    @staticmethod
    def _build_goal_string(task: TravelPlannerTask) -> str:
        """Build goal string from task."""
        parts = [task.query]
        if task.local_constraint:
            parts.append(f"\nConstraints: {json.dumps(task.local_constraint)}")
        if task.budget:
            parts.append(f"\nBudget: ${task.budget}")
        parts.append(f"\nDates: {task.date}")
        parts.append(f"\nPeople: {task.people_number}")
        parts.append(f"\nOrigin: {task.org}")
        parts.append(f"\nDestination: {task.dest}")
        parts.append(f"\nDays: {task.days}")
        return "\n".join(parts)
