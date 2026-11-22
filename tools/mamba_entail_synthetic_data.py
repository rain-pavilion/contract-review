"""Grouped synthetic weak-supervision pairs; never independent human gold.

Every article below is an invented contractual norm, not a current statute.
The source/template family (including all labels and wording variants) stays
in one split. Shared general Chinese scaffolds remain; these splits are not
a certification of legal generalization or an independent benchmark.
"""
from __future__ import annotations
import hashlib
import json
import random
from collections import Counter

LABELS = ("entail", "contradict", "neutral")
# topic, synthetic norm, entailing clause, incompatible clause, unrelated clause
FAMILIES = [
 ("付款期限", "买方应在验收后十日内支付全部价款。", "买方应在验收后十日内付清价款。", "买方仅须在验收后第三十日支付价款，前十日无需付款。", "本合同附件统一使用中文编写。"),
 ("交货数量", "供方必须交付一百台设备，少于一百台不算足额。", "供方交付一百台设备才算足额交货。", "供方仅交付六十台设备即视为足额交货。", "双方各自保存经办人员的联系地址。"),
 ("保密期限", "接收方应在合同终止后三年内继续保密。", "合同终止后三年内接收方仍负保密义务。", "合同终止后三十日保密义务即全部终止。", "本项目使用蓝色文件夹存放纸质资料。"),
 ("验收时限", "买方应在收到设备后七日内完成验收。", "买方收货后七日内完成验收。", "买方可以在收货第六十日才开始验收，前七日不必验收。", "设备铭牌采用黑色印刷文字。"),
 ("通知形式", "解除通知必须采用书面形式，口头通知不生效。", "双方只能用书面通知有效解除合同。", "解除合同只需口头通知，不需要任何书面文件。", "双方办公时间为工作日上午九点至下午五点。"),
 ("所有权保留", "全部价款付清前，设备所有权仍属于卖方。", "买方未付清价款时设备所有权不转移。", "设备一经交付所有权即归买方，无论价款是否付清。", "设备包装箱外应印有运输方向标识。"),
 ("借款期限", "借款人必须在放款后十二个月内归还全部本金。", "借款人在放款后十二个月内归还全部本金。", "放款后第二十四个月才应还本，十二个月内无需归还。", "双方约定借款文件使用双面打印。"),
 ("利息上限", "本约定的年利率不得超过百分之三。", "借款年利率最高为百分之三。", "借款年利率固定为百分之八且必须全部支付。", "借款合同正本由双方分别保管一份。"),
 ("争议地点", "争议只能提交上海仲裁委员会仲裁。", "发生争议时仅提交上海仲裁委员会仲裁。", "一切争议只能提交北京仲裁委员会仲裁。", "双方每季度交换一次项目进度资料。"),
 ("转让同意", "乙方转让合同义务前必须取得甲方书面同意。", "乙方应先取得甲方书面同意才能转让合同义务。", "乙方可直接转让合同义务，无需甲方同意。", "双方项目文件的编号采用年度流水号。"),
 ("分包限制", "承包人不得将全部工程转包给第三人。", "承包人不能把全部工程转包给其他单位。", "承包人可以把全部工程直接转包给任意第三人。", "工程现场的临时会议室位于东侧。"),
 ("危险货物", "运输人收到危险货物前必须取得危险说明书。", "运输人应先取得危险说明书再接收危险货物。", "运输人接收危险货物完全不需要危险说明书。", "运输单据的页边距统一设置为两厘米。"),
 ("质量保修", "卖方应提供自验收起二十四个月的免费保修。", "设备验收后卖方免费保修二十四个月。", "卖方不提供任何免费保修，所有维修均收费。", "买方的仓库编号为三号仓库。"),
 ("提前通知", "单方终止服务须提前三十日书面通知对方。", "终止服务的一方应提前三十日发出书面通知。", "一方可当日终止服务，不需提前通知。", "双方应保持日常联系人电话号码畅通。"),
 ("付款币种", "所有价款必须以人民币结算，不接受其他币种。", "双方仅以人民币结算合同价款。", "全部价款仅以美元结算，不接受人民币。", "合同签署地点为供应方办公室。"),
 ("文件份数", "双方必须各持两份签字合同正本。", "甲乙双方分别保存两份签字合同正本。", "双方各持一份正本即可，无需保存第二份。", "合同总价在双方盖章后保持不变。"),
 ("设备保险", "租赁期间出租人必须为设备投保财产保险。", "出租人在整个租赁期间为设备维持财产保险。", "出租人无须为租赁设备投保任何财产保险。", "租赁设备的安装地点由双方另行确认。"),
 ("交付地址", "所有货物必须交付至南京指定仓库。", "全部货物应运送至南京指定仓库。", "所有货物只交付杭州仓库，不交付南京仓库。", "双方发票联系人在订单附件中列明。"),
 ("数据删除", "服务结束后七日内服务方必须删除全部客户数据。", "服务方应在服务结束后七日内删除全部客户数据。", "服务方可永久保留全部客户数据且无需删除。", "项目会议记录由双方项目经理签字。"),
 ("成果归属", "本项目全部定制软件著作权归甲方所有。", "甲方拥有本项目全部定制软件著作权。", "本项目全部定制软件著作权仅归乙方所有。", "双方通过电子邮件安排例行项目会议。"),
 ("发票开具", "卖方须在每次收款后五日内开具合法发票。", "卖方每次收到款项后五日内向买方开具合法发票。", "卖方收款后无需开具任何发票。", "买方的项目名称应印在设备外箱上。"),
 ("资料公开", "未经乙方书面许可，甲方不得公开乙方技术资料。", "甲方只有取得乙方书面许可后才能公开乙方技术资料。", "甲方可以随时公开乙方技术资料，无需取得许可。", "双方在每月第一个工作日核对工时。"),
 ("报销票据", "申请报销时必须提交原始票据，没有票据不予报销。", "报销申请必须附上原始票据。", "报销完全不需要任何票据，空白申请也必须支付。", "项目负责人应每周更新任务计划。"),
 ("定期报告", "服务方必须每月向客户提交书面进度报告。", "服务方每个月均向客户提供书面进度报告。", "服务方不需提交任何月度进度报告。", "客户的收件邮箱在通讯录中登记。"),
]


def build_dataset(seed=20261114):
    groups = [f"synthetic-family-{i:02d}" for i in range(len(FAMILIES))]
    shuffled = groups[:]
    random.Random(seed).shuffle(shuffled)
    split_groups = {"train": shuffled[:18], "valid": shuffled[18:21], "test": shuffled[21:]}
    group_split = {group: split for split, members in split_groups.items() for group in members}
    rows = []
    for index, (topic, norm, positive, negative, unrelated) in enumerate(FAMILIES):
        group = groups[index]
        for label, premise in zip(LABELS, (positive, negative, unrelated)):
            variants = [premise, f"就{topic}事项，{premise}", f"双方确认：{premise}", f"本协议记载：{premise}"]
            for variant, text in enumerate(variants):
                rows.append({"pair_id": f"{group}-{label}-{variant}", "clause": text,
                    "article": "合成规范：" + norm, "label": label, "source_id": group,
                    "template_family": group, "group_id": group, "topic": topic,
                    "split": group_split[group], "supervision": "synthetic_weak_supervision_not_human_gold"})
    result = {"rows": rows, "group_split": group_split,
              "limitation": "Invented norms and family-grouped synthetic weak labels; generic language scaffolds remain shared. Not independent human gold or proof of current law."}
    for split in split_groups:
        result[split] = [row for row in rows if row["split"] == split]
        assert set(Counter(row["label"] for row in result[split])) == set(LABELS)
    assert len({(row["clause"], row["article"]) for row in rows}) == len(rows) == 288
    for field in ("source_id", "template_family", "group_id"):
        sets = {split: {row[field] for row in result[split]} for split in split_groups}
        assert all(not sets[a] & sets[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test")))
    result["counts"] = {split: dict(Counter(row["label"] for row in result[split])) for split in split_groups}
    canonical = "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n"
    result["sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    return result


if __name__ == "__main__":
    data = build_dataset()
    print(json.dumps({"pairs": len(data["rows"]), "counts": data["counts"], "sha256": data["sha256"]}, ensure_ascii=False, indent=2))
