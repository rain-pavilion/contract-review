"""Generate a source-audited SYNTHETIC UNLABELED annotation candidate pack.

No rules, ReviewEngine, model predictions, answers or invented reviewers are used.
Slots A/B identify separate blank assignments, never actual people.
"""
from __future__ import annotations
import argparse, datetime, json
from pathlib import Path
from acceptance_annotations import SCHEMA, read_rows, validate
from acceptance_common import ROOT, digest, write_json

# Newly authored synthetic cases. Variations are candidate text, not annotations.
CASES = [
('租赁','办公场地','甲方将示例城市明澄园A座301室出租给乙方办公使用，建筑面积约120平方米。附属设施以双方签署的交付清单为准。',
 '租赁期自2027年1月1日至2028年12月31日，甲方于2026年12月25日前交付。乙方需要续租的，应于届满前60日提出书面申请。',
 '月租金8000元，每季度首月5日前支付。保证金16000元，合同结束且完成结算后20个工作日内返还；租金以银行转账记录为支付凭证。',
 '日常小修由乙方承担；因建筑主体或原有管线问题产生的修理由甲方安排。乙方装修前应提交方案并取得甲方书面确认。',
 '任何一方要求提前解除，应提前30日通知对方。乙方拖欠租金超过15日的，甲方可以书面催告；催告后10日仍未补缴的，可解除合同。'),
('租赁','住宅短租','甲方出租示例城市南栖路示例苑2栋502室供乙方居住。房屋交付时，双方记录家具、电器、水电表读数及现有损坏情况。',
 '租期六个月，自2027年3月1日至2027年8月31日。续租事宜由双方于租期届满前另行协商。',
 '月租金3200元，乙方按月预付。押金6400元。正常使用产生的水电燃气费由乙方支付，物业服务费由甲方支付。',
 '甲方因维修需要进入房屋的，原则上提前一日通知乙方；紧急漏水或火灾等情况除外。乙方未经同意不得转租。',
 '乙方提前退租的，已支付租金按实际占用天数结算，押金不予退还。甲方提前收回房屋的，应提前10日通知乙方。'),
('租赁','仓储场地','甲方向乙方提供示例工业园C区3号库房，面积约600平方米，用于存放包装材料。双方应在交付前确认消防设施、通道和用电负荷。',
 '租期自2027年2月1日至2028年1月31日。甲方在2027年2月10日前完成场地交付，乙方按实际交付日期开始使用。',
 '合同列示全年租金180000元。乙方签约时支付100000元，于2027年8月1日前支付90000元；甲方按收到款项开具相应票据。',
 '乙方不得存储易燃易爆或法律禁止存储的物品。甲方负责公共区域维护，乙方负责库房内部作业安全。发生事故应及时通知并保存证据。',
 '因政府依法征收导致无法继续履行的，双方结算已使用期间费用；补偿分配另行协商。因乙方用途不符合约定造成损失的，由乙方承担相应责任。'),
('技术开发','管理系统','乙方受甲方委托开发采购管理系统，包括供应商档案、采购申请、订单和报表四个模块。具体功能、接口、性能要求以双方确认的需求清单附件一为准。',
 '项目自2027年1月10日启动，2027年4月30日前提交测试版本，2027年5月31日前提交验收版本。甲方应在每次交付后10个工作日内书面反馈。',
 '开发费用合计120000元，签约后支付36000元，测试版本交付后支付48000元，最终验收后支付36000元。新增需求由双方另行确认工作量和费用。',
 '乙方修复与需求清单不一致的问题后重新提交验收。源代码、部署文档和管理员账号在尾款支付后移交。双方各自原有技术归原权利人。',
 '项目成果的著作权由双方共有，甲方可在本单位业务中使用。任一方对外许可项目成果，应事先取得另一方书面同意。'),
('技术开发','移动应用','乙方为甲方开发活动报名移动应用，包含实名注册、报名、签到和通知功能。甲方负责提供界面文案及业务规则，乙方负责开发、测试和部署。',
 '开发周期为60个自然日，自甲方确认原型且首付款到账次日起计算。甲方变更已确认原型导致返工的，双方书面确定新的交付日。',
 '项目总费用68000元，分两次各支付34000元。乙方提交可运行安装包即视为完成全部交付，甲方应在三个工作日内支付剩余费用。',
 '甲方未在安装包提交后两日内提出异议的，视为全部验收通过。乙方提供三个月技术支持，超过该期间的维护另行收费。',
 '应用运行收集的用户资料由甲方管理。乙方为定位故障可以调取生产数据，调取方式和范围由项目联系人通过即时通讯确认。'),
('技术开发','数据分析平台','乙方开发销售数据分析平台，甲方提供经脱敏的示例数据和字段说明。交付内容包括数据导入、指标计算、可视化看板及操作手册。',
 '双方将项目划分为原型确认、功能测试、试运行三个里程碑；计划在2027年6月30日前完成。甲方提供资料延迟超过五日，双方另行书面调整时间。',
 '合同价款96000元，首次付款28800元，中期付款38400元，验收付款28800元。甲方收到合法票据后十个工作日内支付对应款项。',
 '乙方应在验收后提供十二个月缺陷修复服务。因第三方接口规则改变产生的改造，由双方确认费用；第三方组件使用情况在交付清单中列明。',
 '乙方保留通用工具及既有代码权利，定制部分在结清款项后授权甲方永久使用。甲方要求取得完整源代码及对外许可权的，由双方另签附件。'),
('买卖','办公服务器','乙方向甲方销售两台示例S型服务器，每台配置双处理器、128GB内存和四块企业级硬盘。品牌、配置、序列号与附件以交货单列明的信息为准。',
 '乙方于2027年2月20日前送达甲方指定地址。甲方在到货后三个工作日内核对数量和外观，在十五日内完成上电测试并出具验收记录。',
 '含税总价78000元，甲方签约后支付30%，验收通过后支付70%。运输及装卸费用包含在价款中。发票信息由甲方书面提供。',
 '乙方保证交付全新产品，并提供三年厂家质保凭证。发生硬件故障，乙方在接到通知后一个工作日内响应，具体修复时间以厂家检测结果为准。',
 '货物在甲方签收前发生毁损的风险由乙方承担。因甲方不当操作造成的损坏不属于免费保修范围，但乙方应协助联系维修。'),
('买卖','耗材分批采购','乙方向甲方分批供应示例办公耗材，商品名称、规格、单价及预计数量见附件目录。目录数量为估计量，实际采购以甲方确认的订单为准。',
 '框架合同有效期为2027年1月1日至2027年12月31日。乙方收到订单后两个工作日内确认是否有货，确认后五个工作日内完成交货。',
 '甲方按月结算，在每月对账单确认后30日内付款。年度预计采购金额100000元，该金额不构成甲方最低采购承诺。',
 '甲方收货时核验数量和包装；质量异议应在收到货物后七日内提出。乙方可根据市场价格变化调整目录单价，调整后立即适用于未结算订单。',
 '乙方延迟交货的，每日按该订单金额0.5%支付违约金；甲方延迟付款的，按照双方另行协商的标准处理。'),
('买卖','设备定金采购','甲方向乙方采购一套示例包装设备，额定速度、尺寸范围、电源及安全要求在技术附件中列明。乙方负责送货、安装和首次操作培训。',
 '乙方应在收到首笔款项后45日内交付，安装调试应在设备到场后十日内完成。验收采用双方确认的连续试运行记录。',
 '总价200000元。甲方签约后支付60000元定金，发货前支付100000元，验收后支付40000元。全部价款中包含运输和安装费用。',
 '设备达不到技术附件约定要求的，乙方应在十五日内采取改进措施，改进后再次组织验收。连续两次未通过的，双方协商更换设备或解除合同。',
 '设备所有权在全部价款付清后转移。交付后因甲方保管不善造成损坏的，由甲方承担。质保期自首次通电之日起十二个月。'),
('服务','信息系统运维','乙方为甲方现有业务系统提供巡检、备份检查、故障定位和版本发布支持。服务资产清单及联系人信息以附件为准，不包含新增功能开发。',
 '服务期自2027年1月1日至2027年12月31日。普通问题在一个工作日内响应，严重影响业务运行的故障在一小时内响应并持续通报处理进度。',
 '年度服务费48000元，按季度支付，每季度12000元。每季度结束后乙方提交服务记录，甲方确认后十个工作日内付款。',
 '乙方仅在甲方授权窗口内操作生产系统，重要操作应保留日志。涉及账号和数据的资料不得用于本合同以外用途，终止后应返还或删除。',
 '因未经授权操作造成直接损失的，责任方承担经核实的相应损失；因不可归责于双方的网络中断导致服务受影响的，双方协商调整服务安排。'),
('服务','品牌设计','甲方委托乙方设计示例品牌标识及基础视觉规范，成果包括标识文件、颜色规范、字体建议和三款应用示例。甲方提供品牌定位及使用场景。',
 '乙方签约后十五日内提供三套初稿，甲方在五日内选择一套继续深化。合同内含两轮修改，超过约定范围的新增设计由双方书面确认。',
 '服务费30000元，签约后支付15000元，最终方案确认后支付15000元。甲方改变已确认方向并要求重新设计的，双方协商增补费用。',
 '最终定稿的可编辑源文件在尾款支付后交付。乙方不得将甲方提供的保密资料对外公开；乙方可以将全部草稿及最终成果用于公开展示。',
 '甲方负责对最终标识进行商标注册检索。乙方保证其交付的原创部分不是直接复制第三方作品，第三方授权字体和图片应另行列明。'),
('服务','管理咨询','乙方为甲方提供仓储流程调研及优化建议，开展两次现场访谈并提交现状分析、改进建议和实施路线三项成果。具体范围见双方确认的工作说明。',
 '项目计划于2027年4月1日至2027年5月15日实施。甲方负责协调访谈人员并提供真实、必要的业务资料，乙方按周提交进度说明。',
 '含税服务费56000元，签约支付28000元，最终报告提交后支付28000元。经甲方预先同意的跨城差旅按实际发生额另行报销。',
 '乙方提出的建议由甲方自行决定是否实施。合同不承诺特定经营收益。乙方应对接触的经营信息保密，保密义务在合同终止后继续有效两年。',
 '最终报告提交至甲方联系人邮箱即为送达。甲方在十个工作日内组织评审，并一次性提供书面意见；双方根据意见确认是否需要补充说明。'),
('劳动','产品助理','甲方聘用乙方担任产品助理，工作地点为示例城市东禾办公区。岗位职责包括需求整理、会议记录和产品测试；岗位调整由双方协商确认。',
 '合同期限自2027年2月1日至2028年1月31日，约定试用期三个月。甲方实行标准工时制度，依法安排休息和休假。',
 '基本工资每月6000元，绩效工资按依法制定并向乙方公示的制度执行，每月10日前支付上月工资。甲方依法为乙方办理社会保险。',
 '乙方入职时参加信息安全培训，对工作中知悉的客户资料承担保密义务。甲方提供必要劳动条件，并依法开展职业健康和安全培训。',
 '解除或终止合同按照法律规定及本合同约定办理，乙方完成工作交接后，甲方出具相关证明。双方发生争议，先行协商，协商不成依法处理。'),
('劳动','技术支持','甲方聘用乙方从事技术支持工作，可根据客户服务计划安排在示例城市各服务点工作。跨城市长期工作需双方另行书面确认。',
 '合同期三年，自2027年3月1日至2030年2月28日，试用期六个月。工作时间原则上每周五日，每日八小时，紧急故障支持按制度安排值班。',
 '月工资8000元，其中基本工资6500元、岗位工资1500元。甲方每月15日前支付。加班事项及相应工资或调休按照法律规定办理。',
 '甲方为乙方提供专项技术培训的，培训费用、服务期及相应责任由双方另行签署协议。乙方不得将客户账号和设备资料用于私人业务。',
 '乙方离职应提前三十日以书面形式通知甲方；试用期内提前三日通知。乙方解除合同的，应向甲方支付固定离职费用20000元。'),
('劳动','行政专员','甲方聘用乙方担任行政专员，主要从事文件管理、会议支持和固定资产登记。工作地点、直属负责人及岗位说明由甲方书面告知。',
 '双方订立无固定期限劳动合同，自2027年1月5日起履行，不再另设试用期。工作时间和休假按照甲方依法制定并公示的制度执行。',
 '乙方月工资5500元，甲方于每月8日前支付，依法代扣代缴个人应承担税费。因岗位发生变化调整工资，应与乙方协商并以书面形式确认。',
 '甲方依法为乙方办理社会保险，乙方配合提供必要资料。与工作有关的个人信息仅用于人事管理及法律规定用途，接触人员应妥善保管。',
 '甲方可以根据经营需要随时解除合同，乙方不得提出任何补偿要求。其他未尽事项按法律法规及双方依法确认的规章制度执行。'),
('承揽','办公室装修','乙方承揽甲方示例办公场所装修，包括墙面、照明和会议区改造。施工图、材料清单和工程量表为合同附件，乙方应按确认版本施工。',
 '施工期自2027年2月10日至2027年3月25日。甲方应提供具备施工条件的场地；需变更方案的，双方先确认变更单及对工期的影响。',
 '暂定含税工程价款180000元。进场后支付54000元，隐蔽工程验收后支付72000元，竣工验收后支付45000元，余款9000元在质保期届满后支付。',
 '乙方对材料进场、隐蔽工程和竣工验收留存记录。甲方在收到验收申请后七日内组织验收，发现问题应书面列明，乙方按确认期限整改。',
 '乙方负责施工现场安全管理和其工作人员管理。甲方提前使用部分区域的，不免除乙方对未完成项目的继续施工及约定质量责任。'),
('承揽','设备安装','乙方负责将甲方采购的示例输送设备安装到指定生产线，包括基础固定、接线、联调和试运行。设备采购质量责任不在本合同安装服务范围内。',
 '计划2027年5月1日进场，2027年5月20日完成。甲方应提前提供供电、基础尺寸和现场作业条件。因现场条件不足无法施工的，乙方及时书面通知。',
 '安装服务费42000元，进场前支付21000元，试运行通过后支付21000元。超出原图纸范围的变更作业应先确认计价依据。',
 '乙方按安装技术要求实施并提交测试记录。未经甲方书面同意，不得将主要安装工作交由第三方完成。乙方提供六个月安装质量保证。',
 '甲方在收到完成通知后应于五日内组织验收。甲方未按期验收的，乙方可以单方确认验收合格并要求立即支付全部尾款。'),
('承揽','展陈制作','甲方委托乙方制作示例展览展示装置，包括展板、灯箱和一套互动装置。尺寸、颜色、材料及图文文件以双方签署确认的制作单为准。',
 '乙方于2027年7月5日前完成制作并运输至展馆，于2027年7月8日前完成安装。展览计划2027年7月1日开幕，现场协调时间由甲方通知。',
 '总价88000元，制作前支付44000元，到场安装完成后支付35200元，撤展并结清物料后支付8800元。新增数量按双方确认单价另行结算。',
 '甲方负责提供图文资料并保证其依法可用，乙方负责制作工艺和安装安全。样品确认后，如现场制作与样品存在差异，乙方应按确认要求调整。',
 '展览结束后乙方负责拆除和清运，其工作不得损坏展馆原有设施。可重复使用部件的归属以制作单为准，未列明的由双方另行协商。'),
]

def records():
    now=datetime.datetime.now(datetime.timezone.utc).isoformat()
    rows=[]
    for i,(kind,topic,subject,period,payment,performance,termination) in enumerate(CASES,1):
        cid=f'SYN-{i:03d}'
        text=(f'{topic}合同\n'
            '候选文本声明：本合同为本次新编的合成材料，主体、地点、业务情境均为示例，不对应真实交易。\n'
            '甲方：示例甲方（合成主体）\n乙方：示例乙方（合成主体）\n'
            f'第一条 合同事项\n{subject}\n第二条 期限与进度\n{period}\n'
            f'第三条 价款与支付\n{payment}\n第四条 履行与交付\n{performance}\n'
            f'第五条 责任与终止\n{termination}\n'
            '第六条 通知及文本\n双方使用合同签署页载明的电子邮箱和通讯地址发送履约通知，联系方式变化应及时书面通知对方。'
            '本合同附件经双方确认后与正文共同构成约定内容，附件调整应保存版本及确认记录。\n'
            '第七条 争议处理\n履行中发生争议，双方先协商解决；协商不成的，依法向有管辖权的机构申请处理。\n'
            '签署页\n甲方签署：________ 乙方签署：________\n签署日期：2026年12月20日\n')
        rows.append({'schema_version':SCHEMA,'id':cid,'text':text,'text_sha256':digest(text),
            'synthetic_candidate':True,'contract_type':kind,
            'source':{'origin':'newly_authored_synthetic_candidate_v1','path':f'候选文本/{cid}.txt',
                'group_id':'synthetic_template_family_'+kind,'license_or_permission':'本次新编合成候选，仅用于项目标注流程与后续明确标为合成的评测；非公开来源真实合同',
                'collected_at':now,'public_source_url':None},
            'status':'unlabeled','split':'unassigned','annotations':[],'reviews':[]})
    return rows

def write_jsonl(path,rows):
    path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in rows),encoding='utf-8')

def create_pack(folder):
    if folder.exists():raise ValueError('refuse overwrite existing annotation pack; use a new version path')
    (folder/'候选文本').mkdir(parents=True)
    rows=records()
    for row in rows:(folder/row['source']['path']).write_text(row['text'],encoding='utf-8')
    write_jsonl(folder/'candidates.jsonl',rows)
    for slot in ('A','B'):
        assignments=[]
        for row in rows:
            assignments.append({**row,'blind_assignment':{'slot':slot,'reviewer_id':None,'started_at':None,'completed_at':None,
                'instructions':'独立阅读原文后人工标注；不得读取另一路标注、仲裁意见或系统预测；完成前保持unlabeled。'}})
        write_jsonl(folder/f'盲标任务_{slot}.jsonl',assignments)
    adjudication=[{**row,'adjudication':{'adjudicator_id':None,'adjudicated_at':None,'rationale':None,'annotations_sha256':None},
        'adjudication_task':{'status':'awaiting_two_independent_manual_reviews','review_A_file':'盲标任务_A.jsonl',
        'review_B_file':'盲标任务_B.jsonl','instructions':'两路人工标注都完成后，保留各自annotations及hash再仲裁。不得用项目规则补标。'}} for row in rows]
    write_jsonl(folder/'仲裁任务_空模板.jsonl',adjudication)
    (folder/'README.md').write_text(README,encoding='utf-8')
    files=[p for p in sorted(folder.rglob('*')) if p.is_file()]
    manifest={'schema_version':'synthetic-candidate-pack-v1','created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status':'unlabeled','gold_ready':False,'synthetic':True,'independently_sourced_real_contracts':False,
        'records':len(rows),'contract_types':sorted({r['contract_type'] for r in rows}),'source_groups':6,
        'annotations_provided':False,'system_predictions_provided':False,'real_reviewers_assigned':False,
        'template_slots':['A','B'],'files':[{'path':p.relative_to(folder).as_posix(),'sha256':digest(p.read_bytes()),'bytes':p.stat().st_size} for p in files]}
    write_json(folder/'manifest.json',manifest);return check_pack(folder)

def check_pack(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8-sig'));errors=[]
    for item in manifest['files']:
        path=folder/item['path']
        if not path.is_file() or digest(path.read_bytes())!=item['sha256']:errors.append('file hash mismatch: '+item['path'])
    rows=read_rows(folder/'candidates.jsonl');base={r['id']:r for r in rows}
    for name in ('candidates.jsonl','盲标任务_A.jsonl','盲标任务_B.jsonl','仲裁任务_空模板.jsonl'):
        current=read_rows(folder/name);report=validate(current)
        if report['status']!='passed':errors.extend(name+': '+e for e in report['errors'])
        if report['gold_ready'] or any(r['status']!='unlabeled' or r['annotations'] or r['reviews'] for r in current):errors.append(name+': must remain wholly unlabeled')
        if {r['id'] for r in current}!=set(base):errors.append(name+': IDs differ')
        for row in current:
            if row['text_sha256']!=base[row['id']]['text_sha256']:errors.append(name+': candidate source changed')
    rejection=validate(rows,require_gold=True)
    if rejection['gold_ready'] or not rejection['errors']:errors.append('unlabeled freeze rejection guard failed')
    return {'status':'passed' if not errors else 'failed','package_status':'unlabeled','gold_ready':False,
        'records':len(rows),'contract_types':sorted({r['contract_type'] for r in rows}),
        'independent_blank_assignment_files':2,'adjudication_blank_files':1,'freeze_unlabeled':'rejected_as_expected',
        'errors':errors,'manifest_sha256':digest((folder/'manifest.json').read_bytes())}

README='''# 合成合同待标注包 第一版

本包包含18份本次新编合成合同候选，覆盖租赁、技术开发、买卖、服务、劳动、承揽六类，每类3份。全部主体和交易情境均为示例。它们不是独立采集的真实合同，不是金标，不附规则答案、风险提示、系统预测或预填人工结论。即使将来完成标注，也应明确称为“人工标注的合成候选集”，不能改称真实业务独立测试集。

## 分发与盲标

1. 管理员核对manifest文件hash，保留只读原始版本，将盲标任务_A.jsonl与盲标任务_B.jsonl分别发给两位真实标注人员。A/B仅为分发槽位，不是虚构的人名。当前reviewer_id为空，未分配任何实际标注人。
2. 两位人员各自只接触自己的文件和原文，独立标注，不查看对方标签、仲裁文件、项目规则输出或模型预测。约定统一标签手册后再开始，手册不得包含本包逐条答案。
3. 标注使用原文Unicode字符下标[start,end)，保留精确evidence。任务类型为entity、risk或law_reference；label由项目预先冻结的标签手册规定。不要把猜测写成已证实事实，无法判断的情况记录notes交仲裁。每份文本允许标注结果为空，但必须确实完成通读与复核。
4. 各自填写真实reviewer_id、时间和独立annotations。独立结果使用UTF-8 JSON；计算annotations SHA-256时，使用json.dumps(...,ensure_ascii=False,sort_keys=True,separators=(',',':'))后UTF-8编码。保留两路原件，不相互覆盖。
5. 管理员收齐后才将两路独立结果交仲裁。仲裁人在仲裁任务_空模板.jsonl的reviews中保存两份完整独立annotations、各自hash、manual方法、真实身份和时间、used_system_predictions=false；再形成最终annotations并填写adjudication真实身份、时间、理由及最终hash。原文与source信息不变。
6. 只有全部完成的记录才可改为adjudicated；未经标注继续保持unlabeled。禁止利用规则或模型自动填标签来消除空白。

## 划分冻结与防泄漏

同一合同类型的3份候选共用source.group_id，按模板家族整体划分，不能拆到train与test两边。先登记项目所有训练/开发来源的完整清单，再通过acceptance_annotations.py freeze检查；空训练清单会被拒绝。清单至少应包含text及source.group_id，外部数据须保留来源与权限。

现有工具检查原文hash、归一化重复、同源分组、训练清单重叠和人工审计链。管理员仍须做近重复/模板变体人工复查，不能仅凭hash声称完全无泄漏。冻结test后不得用于阈值、提示词、标签定义或模型训练调优；任何修改必须生成新版本并重新审查。合成候选在来源独立性上的限制始终保留。

本包当前4个JSONL（候选、A、B、仲裁）全部为unlabeled，annotations与reviews均为空，gold_ready=false。验证工具会加载全部文件、核对hash与IDs，并确认未标注数据无法冻结为金标。

```powershell
python tools/acceptance_annotation_pack.py check --dir docs/验收/待标注包_v1
python tools/acceptance_annotations.py validate --data docs/验收/待标注包_v1/candidates.jsonl --out tools/acceptance_results/annotation_pack_validation.json
```
'''

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['create','check']);p.add_argument('--dir',type=Path,default=ROOT/'docs/验收/待标注包_v1');p.add_argument('--out',type=Path,default=ROOT/'tools/acceptance_results/annotation_pack.json');a=p.parse_args()
    result=create_pack(a.dir) if a.action=='create' else check_pack(a.dir)
    write_json(a.out,result);print(json.dumps(result,ensure_ascii=False));return 0 if result['status']=='passed' else 1
if __name__=='__main__':raise SystemExit(main())
