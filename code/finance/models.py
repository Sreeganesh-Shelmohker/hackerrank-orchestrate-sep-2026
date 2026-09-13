"""
Domain models for Buy or Wait? Finance Engine.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any

@dataclass
class Request:
    request_id: str
    user_id: str
    request_date: str
    request_type: str
    requested_amount: float
    desired_completion_date: str
    allows_partial_payment: bool
    request_text: str

@dataclass
class FinancialProfile:
    user_id: str
    home_currency: str
    current_available_balance: float
    minimum_balance_to_keep: float
    financial_priorities: str
    expense_categories_to_protect: List[str]
    expense_categories_user_is_willing_to_reduce: List[str]
    expense_categories_user_is_willing_to_stop: List[str]
    payment_methods_user_will_consider: List[str]
    max_installment_months: Optional[float]

@dataclass
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str
    amount: Optional[float]
    currency: str
    event_date: str
    settlement_date: str
    status: str
    linked_event_id: Optional[str]
    flexibility: str
    minimum_allowed_amount: Optional[float]

@dataclass
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: str
    payment_amount: float
    number_of_payments: int
    first_payment_date: str
    payment_frequency_days: Optional[float]
    financing_fee: float
    total_payable_amount: float

@dataclass
class NormalizedEvent:
    event_id: str
    user_id: str
    event_type: str
    description: str
    category: str
    direction: str  # 'debit', 'credit', 'non_cash'
    amount: float   # Converted to home_currency
    home_currency: str
    event_date: str
    settlement_date: str
    status: str
    linked_event_id: Optional[str]
    flexibility: str
    minimum_allowed_amount: Optional[float]

@dataclass
class RecurringStream:
    event_id: str
    category: str
    description: str
    amount: float
    day_of_month: int
    flexibility: str
    minimum_allowed_amount: Optional[float]
    is_income: bool
    frequency_days: int = 30  # 30 for monthly, 7 for weekly
    last_settlement_date: str = ""

@dataclass
class FinancialState:
    user_id: str
    home_currency: str
    request_date: str
    starting_available_balance: float  # current_available_balance minus pending debits
    minimum_balance_to_keep: float
    pending_debits_total: float
    recurring_expense_streams: List[RecurringStream] = field(default_factory=list)
    recurring_income_streams: List[RecurringStream] = field(default_factory=list)
    scheduled_future_events: List[NormalizedEvent] = field(default_factory=list)

@dataclass
class ForecastEntry:
    date: str
    starting_balance: float
    income: float
    expenses: float
    plan_payments: float
    ending_balance: float
    is_safe: bool

@dataclass
class SpendingChangeAction:
    action_type: str  # 'stop' or 'reduce_to'
    event_id: str
    new_amount: Optional[float] = None

    def to_string(self) -> str:
        if self.action_type == 'stop':
            return f"stop:{self.event_id}"
        elif self.action_type == 'reduce_to':
            amt_str = f"{self.new_amount:.2f}".rstrip('0').rstrip('.') if self.new_amount is not None else ""
            return f"reduce_to:{self.event_id}:{amt_str}"
        return ""

@dataclass
class CandidatePlan:
    method: str  # 'full_payment', 'partial_payment', 'installments', 'wait', 'not_recommended'
    affordability_status: str  # 'affordable_now', 'affordable_with_plan', 'affordable_later', 'not_affordable'
    payment_plan_str: str
    earliest_date_for_full_payment: Optional[str]
    spending_changes_str: str
    spending_changes: List[SpendingChangeAction]
    total_paid: float
    start_date: str
    num_payments: int
    payment_option_id: Optional[str]
    explanation: str

@dataclass
class RecommendationResult:
    request_id: str
    amount_safe_to_pay: float
    affordability_status: str
    recommended_payment_method: str
    payment_plan: str
    earliest_date_for_full_payment: str
    spending_changes_needed: str
    decision_explanation: str
