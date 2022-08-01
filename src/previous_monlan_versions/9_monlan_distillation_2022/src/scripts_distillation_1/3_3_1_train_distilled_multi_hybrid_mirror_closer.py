
import os
from src.monlan.utils.save_load import *
from src.monlan.modular_agents.DistilledMultiHybridMirrorCloser import DistilledMultiHybridMirrorCloser

composite_agent = load(os.path.join("../../../../../models/", "kaiming_composite_agent_7.pkl"))
closer = DistilledMultiHybridMirrorCloser(buyer_rl_agent=composite_agent.agents["buyer"],
                                          seller_rl_agent=composite_agent.agents["seller"])

buyer_samples = load(os.path.join("../../../../../data/interim", "buyer_samples.pkl"))
seller_samples = load(os.path.join("../../../../../data/interim", "seller_samples.pkl"))

#########
# debug
"""buyer_samples["id"] = buyer_samples["id"][:10000]
buyer_samples["x_raw"] = buyer_samples["x_raw"][:10000]
buyer_samples["y"] = buyer_samples["y"][:10000]
seller_samples["id"] = seller_samples["id"][:10000]
seller_samples["x_raw"] = seller_samples["x_raw"][:10000]
seller_samples["y"] = seller_samples["y"][:10000]"""
#########

closer.fit( buyer_samples, seller_samples,
            ema_norm_n = 40, smooth_target_n = 40,
            test_size=0.1 )

save(closer, os.path.join("../../../../../models", "distilled_multi_hybrid_mirror_closer.pkl"))
print("done")
