
import os
from src.monlan.utils.save_load import *
from src.monlan.modular_agents.DistilledTCNMirrorCloser import DistilledTCNMirrorCloser

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

closer = DistilledTCNMirrorCloser(input_size=7)
closer.fit(buyer_samples, seller_samples,
           epochs = 202, warm_up_epochs = 2, batch_size = 256, learning_rate = 0.001, batch_norm_momentum=0.1,
           checkpoint_save_path = os.path.join("../../../../../models", "best_val_distilled_tcn_mirror_closer.pkl"))

save(closer, os.path.join("../../../../../models", "distilled_tcn_mirror_closer.pkl"))
print("done")
