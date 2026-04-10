import math
import torch
import os

def convert_vrp_to_pt(file_path, output_name=None):
    """
    解析 .vrp 文件并转换成适配 SDVRP / POMO 的 .pt 格式
    - 同时保存原始坐标 (_origin)
    - depot 与 node 同步归一化到 [0,1]
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    capacity = 0
    coords = {}
    demands = {}
    section = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("CAPACITY"):
            capacity = int(line.split(":")[-1] or line.split()[-1])
        elif line.startswith("NODE_COORD_SECTION"):
            section = "COORD"
            continue
        elif line.startswith("DEMAND_SECTION"):
            section = "DEMAND"
            continue
        elif line.startswith("DEPOT_SECTION") or line.startswith("EOF"):
            section = None
            continue

        parts = line.split()
        if section == "COORD":
            node_id = int(parts[0])
            x, y = float(parts[1]), float(parts[2])
            coords[node_id] = [x, y]
        elif section == "DEMAND":
            node_id = int(parts[0])
            demand = float(parts[1])
            demands[node_id] = demand

    # ========= 节点排序 =========
    all_node_ids = sorted(coords.keys())

    # 假设第一个节点是 depot
    depot_id = all_node_ids[0]
    customer_ids = all_node_ids[1:]

    # ========= 原始坐标 (origin) =========
    depot_xy_origin = torch.tensor(
        [coords[depot_id]], dtype=torch.float32
    )  # [1, 2]

    node_xy_origin = torch.tensor(
        [coords[i] for i in customer_ids], dtype=torch.float32
    )  # [N, 2]

    customer_demand = torch.tensor(
        [demands[i] for i in customer_ids], dtype=torch.float32
    ) /capacity # [N]

    # ========= 同步归一化 =========
    all_xy = torch.cat([depot_xy_origin, node_xy_origin], dim=0)  # [N+1, 2]

    xy_min = all_xy.min(dim=0).values
    xy_max = all_xy.max(dim=0).values
    scale = xy_max - xy_min

    # 防止极端情况（所有点在一条线上）
    scale[scale == 0] = 1.0

    depot_xy = (depot_xy_origin - xy_min) / scale          # [1, 2]
    node_xy = (node_xy_origin - xy_min) / scale            # [N, 2]

    # ========= 构造数据集 =========
    dataset = {
        "problem_size": node_xy.size()[0],
        # 归一化后的（RL 使用）
        "depot_xy": depot_xy.unsqueeze(0),          # [1, 1, 2]
        "node_xy": node_xy.unsqueeze(0),            # [1, N, 2]

        # 原始坐标（评估 / 可视化）
        "depot_xy_origin": depot_xy_origin.unsqueeze(0),     # [1, 1, 2]
        "node_xy_origin": node_xy_origin.unsqueeze(0),           # [1, N, 2]

        # 需求与容量
        "node_demand": customer_demand.unsqueeze(0),         # [1, N]
        "se_cars_capacity": 1,
        "car_fleet": 20
    }

    # ========= 保存 =========
    if output_name is None:
        output_name = os.path.splitext(file_path)[0] + ".pt"

    torch.save(dataset, output_name)

    print(f"成功转换并保存至: {output_name}")
    print(f"客户数量: {len(customer_ids)}, 车辆容量: {capacity}")
    print(f"x范围: [{xy_min[0]:.2f}, {xy_max[0]:.2f}], "
          f"y范围: [{xy_min[1]:.2f}, {xy_max[1]:.2f}]")

# === 执行转换 ===
input_file = "A-n32-k5.vrp"
if os.path.exists(input_file):
    convert_vrp_to_pt(input_file)
else:
    print(f"未找到文件: {input_file}")