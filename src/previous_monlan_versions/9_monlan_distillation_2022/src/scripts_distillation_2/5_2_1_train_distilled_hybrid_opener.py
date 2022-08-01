
import os
from src.monlan.utils.save_load import *
from src.monlan.modular_agents.DistilledHybridMirrorOpener import DistilledHybridMirrorOpener

buyer_samples = load(os.path.join("../../../../../data/interim", "distilled_buyer_samples.pkl"))
seller_samples = load(os.path.join("../../../../../data/interim", "distilled_seller_samples.pkl"))
mirror_closer = load(os.path.join("../../../../../models", "best_val_distilled_tcn_mirror_closer.pkl"))

#########
# debug
"""buyer_samples["id"] = buyer_samples["id"][:10000]
buyer_samples["x_raw"] = buyer_samples["x_raw"][:10000]
buyer_samples["y"] = buyer_samples["y"][:10000]
seller_samples["id"] = seller_samples["id"][:10000]
seller_samples["x_raw"] = seller_samples["x_raw"][:10000]
seller_samples["y"] = seller_samples["y"][:10000]"""
#########

closer = DistilledHybridMirrorOpener( mirror_closer )
closer.fit( buyer_samples, seller_samples, ema_norm_n = 20, smooth_target_n = 20, test_size=0.2, show_plots=True )

save(closer, os.path.join("../../../../../models", "distilled_hybrid_opener.pkl"))
print("done")
