"""让节点③的测试夹具跟上收紧后的口径（一次性 patch）。

收紧后的规则：
- 资质只在命中强线索（涉密/等级保护/信息系统集成/信息安全服务/CMMI/网络安全）时建 requires 边；
- 权限只在正文出现"代理/授权/委托/代表人"语境时才检查法定代表人线索。

原夹具写的是"政务数据平台运维服务"，属弱词，收紧后按设计不再报警 —— 这是预期的行为
变化，不是回归。夹具改成强线索表述，并显式说明为什么要这么写。
"""

import pathlib

p = pathlib.Path("zhilv_qiling_v2/tests_v2/test_nodes.py")
text = p.read_text(encoding="utf-8")

old = '''LICENSE_MISSING_CONTRACT = """信息系统运维服务采购合同
采购人（甲方）：某市大数据管理局
供应商（乙方）：云图数据技术有限公司
第一条 服务内容
1.1 乙方为甲方提供政务数据平台运维服务，含机房设备巡检与应急响应。'''
new = '''# 夹具要点：必须写进"强线索"词（信息系统集成/等级保护/涉密），否则节点③按设计不报资质。
# 同时要出现"委托/授权"语境，权限检查才会启动 —— 这两条口径见 subject_graph.py 的注释。
LICENSE_MISSING_CONTRACT = """信息系统集成与等级保护测评服务采购合同
采购人（甲方）：某市大数据管理局
供应商（乙方）：云图数据技术有限公司
第一条 服务内容
1.1 甲方委托乙方承担政务信息系统集成与网络安全等级保护测评，含机房设备巡检与应急响应。'''
assert old in text, "找不到 LICENSE_MISSING_CONTRACT"
text = text.replace(old, new, 1)

p.write_text(text, encoding="utf-8")
print("test_nodes 夹具已更新")

# 让 .gitignore 忽略预训练权重与大体积产物，避免仓库被 3 GB 权重撑爆
gi = pathlib.Path(".gitignore")
gtext = gi.read_text(encoding="utf-8")
add = """
# 预训练权重与 GPU 产物（体积大，用 tools/_dl_chunked.py 现取）
zhilv_qiling_v2/models/pretrained/
zhilv_qiling_v2/models/bert_ner/model.pt
zhilv_qiling_v2/models/bert_ner/
zhilv_qiling_v2/models/bert_entail/
zhilv_qiling_v2/models/bge_m3/
"""
if "models/pretrained/" not in gtext:
    gi.write_text(gtext.rstrip() + "\n" + add, encoding="utf-8")
    print(".gitignore 已忽略权重目录")
else:
    print(".gitignore 已有权重规则")
