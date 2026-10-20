# 分支 pred — 预测编码读出（进行中）

## 背景（2026-10-21 诊断）
- 旧评估在 token 自身窗内第 56 步读出（当前字符已注入），target 又是 ids[t] → 实际是自联想重建；与 LM 不可比。
- 学出的 γ2/γc 读出失效：同快照 P_V 直接解码当前字符 93.7%，γ2 读出仅 10.5%（γ2 零增益、γc 有害）。
- 真实下一字符基线：unigram 382.5、KN bigram 59.8、KN trigram 38.2、transformer(1ep) 44.2。

## 方向
- 评估改 next-char（shift+1）+ 温度校准。
- 读出改为自监督预测编码：状态 t → 下一 token 的 P 脉冲模式（纯自身活动，无 BP、无 t+1 目标注入），NLMS/delta 更新；不覆盖旧文件（版本化文件名）。
- 目标：先把读出做到 trigram 级（PPL<45），再叠加状态特征。

## 状态
骨架已建（br/pred：readout=pred、readout_feat=pv、read_target=pspk、eta_read=NLMS）；3k 测试读出尺度不足，待稀疏高维上下文特征。
