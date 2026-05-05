import sys
import os
import pickle as pkl
import torch
from tqdm import tqdm

sys.path.append(os.getcwd())

from core.utils.pf_losses_utils import PowerBalanceLoss as PBL


root_to_model = {
    "runs/gns_task_1_3": "GNS",
    "runs/canos_task_1_3": "CANOS",
    "runs/pfnet_task_1_3": "PFNet",
}

if __name__ == "__main__":
    gns = torch.load(
        "runs_gns_task_1_3.pkl",
        map_location=torch.device("cpu")
    )
    pfnet = torch.load(
        "runs_pfnet_task_1_3.pkl",
        map_location=torch.device("cpu")
    )
    canos = torch.load(
        "runs_canos_task_1_3.pkl",
        map_location=torch.device("cpu")
    )
    results = {
        "CANOS": canos,
        "PFNet": pfnet,
        "GNS": gns
    }

    per_model = {}
    for model_name, predictions in results.items():
        pbl_maxs = []
        pqs_pbl = []
        pvs_pbl = []
        slacks_pbl = []
        for per_grid_type in predictions.values():
            per_grid_type = per_grid_type[0]
            for pred in tqdm(per_grid_type):
                # CALCULATE PBL MAX
                predictions = pred["predictions"]
                edge_attr = pred["edge_attr"]
                data = pred["data"]

                edge_index = data["bus", "branch", "bus"].edge_index
                shunt_g = data["bus"].shunt[:, 0]
                shunt_b = data["bus"].shunt[:, 1]
                V_pred, theta_pred, Pnet, Qnet = predictions
                r, x, bs, tau, theta_shift = edge_attr
                src, dst = edge_index

                delta_P, delta_Q = PBL.calculate_PBL(
                    V_pred,
                    theta_pred,
                    Pnet,
                    Qnet,  # predictions
                    r,
                    x,
                    bs,
                    tau,
                    theta_shift,  # line attributes
                    shunt_g,
                    shunt_b,  # shunt values
                    src,
                    dst,  # line connections
                )
                delta_PQ_2 = delta_P**2 + delta_Q**2
                delta_PQ_magn = torch.sqrt(delta_PQ_2)
                pbl_maxs.append(delta_PQ_magn.max().item())

                bus_type = data["bus"].bus_type
                pqs = delta_PQ_magn[bus_type == 1].mean()
                pvs = delta_PQ_magn[bus_type == 2].mean()
                slack = delta_PQ_magn[bus_type == 3].mean()

                pqs_pbl.append(pqs.item())
                pvs_pbl.append(pvs.item())
                slacks_pbl.append(slack.item())

        per_model[model_name] = {
            "pbl_maxs": pbl_maxs,
            "pqs_pbl": pqs_pbl,
            "pvs_pbl": pvs_pbl,
            "slacks_pbl": slacks_pbl,
        }

    torch.save(per_model, "values_for_failure_analysis.pkl")
