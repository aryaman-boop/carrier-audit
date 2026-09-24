export type Disposition =
  | "AUTO_APPROVED"
  | "AUTO_DISPUTED"
  | "NEEDS_REVIEW"
  | "BLOCKED_EXTRACTION";

export interface Charge {
  code: string;
  description: string;
  amount: number;
}

export interface ExpectedCharge {
  code: string;
  basis: string;
  amount: number;
}

export interface Recomputation {
  pro_number: string;
  matched_shipment_id: string | null;
  match_method: "PRO_EXACT" | "FUZZY_LANE_DATE_WEIGHT" | "UNMATCHED";
  match_confidence: number;
  expected_charges: ExpectedCharge[];
  expected_total: number;
  billed_total: number;
  variance: number;
  variance_pct: number;
  weight_delta_lb: number | null;
  class_delta: number | null;
  unsupported_accessorials: string[];
  fuel_pct_billed: number | null;
  fuel_pct_expected: number | null;
  is_duplicate_pro: boolean;
}

export interface Clause {
  id: string;
  rule_id: string;
  title: string;
  text: string;
  score: number;
  source: string;
}

export interface AuditedLine {
  pro_number: string;
  invoice_number: string;
  carrier_name: string;
  extracted: {
    pro_number: string;
    bol_number: string | null;
    ship_date: string | null;
    origin_zip: string | null;
    dest_zip: string | null;
    billed_weight_lb: number;
    billed_class: number;
    charges: Charge[];
    line_total: number;
  };
  _invoice_date?: string | null;
  // Days left in the MFS-400 dispute window. Derived per request, never stored --
  // a stored countdown is wrong the next morning.
  _dispute_days_left?: number | null;
  recomputation: Recomputation;
  classification: {
    category: string;
    reasoning: string;
    cited_clause_ids: string[];
  };
  disposition: Disposition;
  policy_reason: string;
  /** "code" when a deterministic short-circuit resolved it, "model" otherwise. */
  decided_by: "code" | "model";
  cited_clauses: Clause[];
  /** Demo-only: present because the data is synthetic and the defect is known. */
  _ground_truth: string | null;
  _ground_truth_note: string | null;
  _decision: { human_action: string; human_category: string | null } | null;
}

export interface Summary {
  run_id: string;
  lines: number;
  by_disposition: Record<string, number>;
  by_category: Record<string, number>;
  automation_rate: number;
  escalation_rate: number;
  total_billed_usd: number;
  recoverable_usd: number;
  human_decisions: number;
  human_override_rate: number | null;
}

export interface GateInfo {
  filename: string;
  sha256: string;
  size_bytes: number;
  gates_passed: string[];
  rejected_at: string | null;
  reason: string | null;
  detail: string | null;
  cost_usd: number;
  accepted: boolean;
}

export interface UploadResult {
  accepted: boolean;
  gate: GateInfo;
  triage?: { is_freight_invoice: boolean; document_kind: string; reason: string };
  ownership?: { carrier_name_matches_contract: boolean; pro_match_rate: number; lines: number; matched: number };
  invoice_number?: string;
  extraction_method?: string;
  run_id?: string;
  lines?: number;
  recoverable_usd?: number;
  by_disposition?: Record<string, number>;
  usage?: { cost_usd: number; calls: number };
}
