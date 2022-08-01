
import os
from src.monlan.utils.save_load import *
from src.monlan.modular_agents.DistilledTCNMirrorOpener import DistilledTCNMirrorOpener

buyer_samples = load(os.path.join("../../../../../data/interim", "distilled_buyer_samples.pkl"))
seller_samples = load(os.path.join("../../../../../data/interim", "distilled_seller_samples.pkl"))

#########
# debug
"""buyer_samples["id"] = buyer_samples["id"][:10000]
buyer_samples["x_raw"] = buyer_samples["x_raw"][:10000]
buyer_samples["y"] = buyer_samples["y"][:10000]
seller_samples["id"] = seller_samples["id"][:10000]
seller_samples["x_raw"] = seller_samples["x_raw"][:10000]
seller_samples["y"] = seller_samples["y"][:10000]"""
#########

closer = DistilledTCNMirrorOpener(input_size=7)
closer.fit(buyer_samples, seller_samples,
           epochs = 402, warm_up_epochs = 2, batch_size = 160, learning_rate = 0.001,
           checkpoint_save_path = os.path.join("../../../../../models", "best_val_distilled_tcn_mirror_opener.pkl"))

save(closer, os.path.join("../../../../../models", "distilled_tcn_mirror_opener.pkl"))
print("done")
