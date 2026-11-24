"""Versioned review questions, separate from execution records and findings."""
from __future__ import annotations

import copy


_SCOPE = "仅针对本次提供的合同文本与证据；外部事实、完整交易背景和最终法律结论仍须核验。"


def _point(point_id, domain, question, criterion, agent, codes, evidence, impact, recommendation):
    return {"point_id": point_id, "domain": domain, "question": question, "criterion": criterion,
        "responsible_agent": agent, "risk_codes": codes, "required_evidence": evidence,
        "scope": _SCOPE, "version": "review-points-v1", "impact_template": impact,
        "recommendation_template": recommendation}


_CATALOG = [
    _point("subject.identity", "subject", "合同相对方是否明确、身份信息是否可核对？",
        "逐一核对签约主体名称、身份识别信息与签署主体；合同自述不代替登记或身份证明核验。", "subject",
        ["party_missing", "missing_party_info"], ["合同主体及签署页", "主体登记或身份证明及来源记录"],
        "主体信息不足或不一致可能导致相对方识别、送达与责任追索发生争议。",
        "补充并核对双方名称、身份识别信息、联系方式及签署主体，保留来源和核验日期。"),
    _point("subject.authorization", "subject", "签约人是否有权代表该主体签署本合同？",
        "核对签约人与主体的关系、授权文件、授权事项、权限及有效期，并对应本合同。", "subject",
        ["need_authorization"], ["签署页", "法定代表人信息或本合同的授权证明"],
        "授权范围或效力不明可能引发合同约束力、代理权限与履行责任争议。",
        "取得可核对的代表身份或签约授权证明，确认签约事项、权限、期限与本合同一致。"),
    _point("subject.qualification", "subject", "履行合同所需的经营资质是否具备且适用？",
        "结合约定业务核对所需资质、持证主体、业务范围及有效期；未获证据的事项保持待核验。", "subject",
        ["need_license"], ["具体业务范围", "相关许可证或资质及来源记录"],
        "资质不足或适用范围不明可能影响业务履行，并产生监管或履约责任争议。",
        "根据实际业务核对许可要求，补充相关资质、有效期、适用范围和独立来源记录。"),
    _point("subject.title", "subject", "处分、出租或交付标的的权属及权限是否有依据？",
        "将合同标的与权属、出租或处分权限证明对应，并核对限制及第三人权利。", "subject",
        ["unverified_property", "need_title_cert"], ["标的明细", "权属证书或处分、出租授权证明"],
        "权属或处分权限未核实可能导致无法交付、第三人主张权利或履行受阻。",
        "补充与本合同标的一致的权属或授权证明，核对第三人权利和处分限制。"),
    _point("commercial.amount", "commercial", "价款、费用与计算口径是否明确且一致？",
        "对应同一标的、期间和条件核对金额、大小写、币种、单位、税费及计算口径；OCR数值须先核对原件。", "logic",
        ["amount_missing", "amount_inconsistent", "fee_mismatch", "abnormal_price", "zero_rent"],
        ["价款及费用条款", "计价明细、计算依据及相关附件"],
        "金额缺漏、口径不一或异常定价可能导致付款金额、税费承担与结算争议。",
        "核对同一事项的大小写金额、单价数量、总额、币种及税费口径，明确适用条件并修正冲突。"),
    _point("commercial.payment_terms", "commercial", "付款义务、起算条件和支付期限是否可以确定？",
        "逐项核对付款主体、金额或计算方式、起算事件、期限与引用条件；有限文本筛查不能证明完整付款安排合理。", "logic",
        [], ["完整付款与结算条款", "付款触发条件、验收约定及被引用附件"],
        "付款条件、期限或引用关系不清可能导致付款时间难以确定、资金占用或履约争议。",
        "将每笔付款与明确的金额或口径、起算条件、期限和所引条款对应，并核对完整交易安排。"),
    _point("performance.delivery", "performance", "交付内容、时间、地点及先后关系是否明确？",
        "核对交付义务与触发条件，检查日期是否缺失、冲突或先后倒置，并区分原合同、变更和补充协议。", "logic",
        ["vague_delivery", "date_missing", "date_reversed"], ["交付与履行期限条款", "相关日程、补充协议及原始页面"],
        "交付安排缺漏或日期冲突可能导致履行标准、迟延责任及履约顺序发生争议。",
        "明确交付内容、地点、时间和触发条件，核对原件日期以及变更约定的适用顺序。"),
    _point("performance.acceptance", "performance", "验收标准与确认机制是否足以确定合格与否？",
        "核对验收标准、方式、期限、异议处理和不合格后果，并检查所引技术附件是否可用。", "logic",
        ["unclear_standard"], ["验收条款", "技术标准、质量约定及验收附件"],
        "验收标准或程序不清可能造成合格认定、付款触发与质量责任争议。",
        "补充可核对的质量和验收标准，明确验收期限、异议通知及不合格后的补救安排。"),
    _point("liability.balance", "liability", "免责、违约责任与单方权利安排是否需重点复核？",
        "结合双方义务和触发条件审查免责、单方变更、权利放弃及违约责任，避免脱离上下文作效力判断。", "logic",
        ["liability_exemption", "one_sided_breach", "waiver_of_rights", "unilateral_modify"],
        ["双方义务与违约责任", "免责、变更及权利放弃条款的完整上下文"],
        "责任安排失衡或权利限制过宽可能扩大一方损失，并引发条款适用或效力争议。",
        "结合完整权利义务核对免责范围与例外，明确违约责任、补救措施和单方权利的适用条件。"),
    _point("legal.compliance", "legal", "所给条款是否存在需要核对法律适用的合规问题？",
        "将具体合同事实与法律规范、适用条件及例外逐项对应；模型或规则提示不能替代法律适用审查。", "statute",
        ["illegal_terms"], ["相关合同条款与交易事实", "可核对的法律文本、适用时间及依据"],
        "合规条件未满足或法律适用有误可能影响履行、条款效力或相关责任承担。",
        "定位具体条款与交易事实，核对规范版本、适用条件和例外，再由复核人记录法律适用意见。"),
    _point("legal.dispute", "legal", "争议解决方式与机构、地点是否明确且可核对？",
        "检查争议解决约定是否缺失、相互冲突或机构不明，并结合相关事实核对管辖或仲裁约定。", "logic",
        ["dispute_resolution_missing", "jurisdiction_ambiguous"], ["争议解决完整条款", "约定机构及相关连接事实"],
        "争议解决约定缺漏或含混可能增加程序争议、时间成本和维权成本。",
        "明确选用的争议解决方式、机构及必要信息，并核对完整约定与相关连接事实。"),
    _point("consistency.references", "consistency", "条款引用、重复和跨条款安排是否存在矛盾？",
        "逐项核对引用对象是否存在、重复条款是否冲突、相关条款能否在相同主体、标的和条件下共同履行。", "logic",
        ["dangling_reference", "duplicate_clause", "cross_clause_inconsistency"], ["被引用条款与附件", "相关条款及修订文件的完整上下文"],
        "引用缺失、重复或跨条款冲突可能导致解释分歧、责任不清或履行障碍。",
        "补齐引用对象，核对重复与冲突条款的适用条件，必要时明确优先顺序和统一表述。"),
    _point("confidentiality", "confidentiality", "需要保护的信息及保密责任是否有相应安排？",
        "结合所给交易信息核对保密范围、使用与披露条件、保护措施、期限和责任；不能仅凭未检出条款认定无义务。", "logic",
        ["confidentiality_missing"], ["交易涉及的信息范围", "保密、披露及资料交付条款"],
        "保密安排缺漏或范围不清可能导致信息泄露、用途失控或责任承担争议。",
        "结合实际敏感信息明确保密范围、允许披露情形、保护措施、期限与违约责任。"),
    _point("subject.scope", "subject", "合同标的、数量及履行范围是否可以确定？",
        "核对标的名称、规格、数量、服务边界和附件，检查是否能据此确定双方具体给付。", "ner",
        ["subject_unspecified"], ["标的或服务范围条款", "规格、数量、工作说明及相关附件"],
        "标的或履行范围不明确可能导致交付边界、价款对应关系与验收要求发生争议。",
        "补充标的名称、规格、数量或服务边界，并将附件与具体履行义务对应。"),
]


def review_point_catalog() -> list[dict]:
    """Return fresh question definitions; catalog presence is not execution."""
    return copy.deepcopy(_CATALOG)
