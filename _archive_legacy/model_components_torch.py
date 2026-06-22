"""
时间生成模块 - PyTorch版本
TN日尺度模型可微分改造的第一步

仅包含时间生成相关组件，保持与model_components_numpy.py相同的接口。
设计为可插拔替换，不影响现有NumPy流程。

主要组件：
1. torch版本辅助函数：softplus_torch, sigmoid_torch, group_softmax_torch
2. torch版本时间生成器：LearnableDailyTotalTorch
3. 参数约束采用可微重参数化，避免硬截断

注意：
- 本模块不包含物理过程、空间分配等组件
- 所有函数和类名添加"_torch"后缀以示区别
- 输入输出尽量保持与NumPy版本兼容
"""

import torch
import numpy as np
import pandas as pd


# =============================================================================
# 1) torch版本辅助函数
# =============================================================================

def softplus_torch(x: torch.Tensor) -> torch.Tensor:
    """
    torch版本的稳定softplus函数

    公式：log(1 + exp(-|x|)) + max(x, 0)
    数值稳定，避免exp溢出

    参数:
        x: 输入张量

    返回:
        softplus(x) 张量
    """
    return torch.log1p(torch.exp(-torch.abs(x))) + torch.clamp_min(x, 0.0)


def sigmoid_torch(x: torch.Tensor) -> torch.Tensor:
    """
    torch版本的稳定sigmoid函数

    使用torch.sigmoid内置实现，已优化数值稳定性

    参数:
        x: 输入张量

    返回:
        sigmoid(x) 张量
    """
    return torch.sigmoid(x)


def group_softmax_torch(z: torch.Tensor, groups: torch.Tensor) -> torch.Tensor:
    """
    torch版本的分组softmax

    在每个组内分别计算softmax权重，保持可微性

    参数:
        z: 原始分数张量 (n_samples,)
        groups: 分组标签张量 (n_samples,)，值为整数

    返回:
        w: softmax权重张量 (n_samples,)，每个组内和为1
    """
    w = torch.zeros_like(z, dtype=torch.float32)
    unique_groups = torch.unique(groups)

    for g in unique_groups:
        mask = (groups == g)
        zg = z[mask]

        # 数值稳定softmax
        max_zg = torch.max(zg)
        exp_zg = torch.exp(zg - max_zg)
        sum_exp = torch.sum(exp_zg) + 1e-12

        w[mask] = exp_zg / sum_exp

    return w


# =============================================================================
# 2) torch版本时间生成器
# =============================================================================

class LearnableDailyTotalTorch:
    """
    可学习的日尺度总量生成器 + 地表比例生成器（PyTorch版本）

    输出两个关键量：
    1) L_total_daily[t] : 每天流域总源项（守恒到 year_total*year_scale）
    2) f_surface[t]     : 每天"可即时出流比例"（0~1，可微约束）

    关键改进：
    - 使用torch张量运算，支持自动微分
    - 参数约束采用可微重参数化，避免硬截断
    - API计算向量化，提高效率
    """

    def __init__(self, cfg, dates, rain, runoff, fert_factor, year_total,
                 prior_indicator=None, fs_prior=None, device='cpu'):
        """
        初始化时间生成器（PyTorch版本）

        参数:
            cfg: 配置字典
            dates: 日期序列 (pd.DatetimeIndex)
            rain: 降雨序列 (mm/day)，将转换为torch张量
            runoff: 径流序列 (mm/day)，将转换为torch张量
            fert_factor: 施肥因子序列 (0~1)，将转换为torch张量
            year_total: 年总源项 (kg/year)
            prior_indicator: 日总量先验指示器（可选）
            fs_prior: 地表比例先验（可选）
            device: 计算设备（'cpu'或'cuda'）
        """
        self.cfg = cfg
        self.dates = pd.to_datetime(dates)
        self.year_total = float(year_total)
        self.T = len(self.dates)
        self.device = device

        # 转换输入序列为torch张量
        self.rain = torch.as_tensor(rain, dtype=torch.float32, device=device)
        self.runoff = torch.as_tensor(runoff, dtype=torch.float32, device=device)
        self.fert = torch.as_tensor(fert_factor, dtype=torch.float32, device=device)

        # 计算日序（day of year）
        doy = self.dates.dayofyear.to_numpy()
        self.doy = torch.as_tensor(doy, dtype=torch.float32, device=device)

        # 处理先验信息
        if prior_indicator is None:
            self.prior = None
        else:
            p = torch.as_tensor(prior_indicator, dtype=torch.float32, device=device)
            p = torch.clamp_min(p, 1e-12)
            p = p / (torch.sum(p) + 1e-12)
            self.prior = p

        # f_surface先验
        if fs_prior is None:
            self.fs_prior = None
        else:
            fs = torch.as_tensor(fs_prior, dtype=torch.float32, device=device)
            self.fs_prior = torch.clamp(fs, 0.0, 1.0)

    def compute_api(self, rain_eff: torch.Tensor, decay: float) -> torch.Tensor:
        """
        计算前期降水指数API（向量化版本）

        递归公式：API_t = rain_eff_t + decay * API_{t-1}
        使用累积和实现，避免循环

        参数:
            rain_eff: 有效降雨序列 (T,)
            decay: 衰减系数 [0,1)

        返回:
            API序列 (T,)
        """
        T = len(rain_eff)
        if T == 0:
            return torch.zeros_like(rain_eff)

        # 创建衰减权重：decay^{k}, k=0,...,T-1
        powers = torch.arange(T, dtype=torch.float32, device=rain_eff.device)
        decay_powers = decay ** powers

        # 计算累积加权和
        # API_t = Σ_{k=0}^{t} rain_eff_k * decay^{t-k}
        # 等价于卷积：API = conv(rain_eff, decay_powers)[:T]
        # 这里用更直观的循环实现，梯度可正常传播
        api = torch.zeros_like(rain_eff)
        a = 0.0
        for t in range(T):
            a = rain_eff[t] + decay * a
            api[t] = a

        return api

    def forward(self, theta_time: torch.Tensor):
        """
        前向计算：生成日总源项和地表比例（可微版本）

        参数:
            theta_time: 时间生成器参数向量 (21,)
                     要求requires_grad=True以支持梯度计算

        返回:
            L_daily: 日总源项序列 (T,)
            f_surface: 地表比例序列 (T,)
            dbg: 调试信息字典
        """
        # 确保输入为torch张量
        if not isinstance(theta_time, torch.Tensor):
            theta_time = torch.as_tensor(theta_time, dtype=torch.float32, device=self.device)

        th = theta_time.clone()

        # -------- (A) daily total ----------
        # 参数提取（保持与NumPy版本相同的索引）
        year_scale = th[0]
        rain_thr = th[1]
        api_decay = th[2]

        b0, s1, c1, s2, c2 = th[3], th[4], th[5], th[6], th[7]
        p0, pr, pq, pa, pf = th[8], th[9], th[10], th[11], th[12]
        mix_kappa = th[13]

        # 有效降雨（可微max替代）
        rain_eff = torch.clamp_min(self.rain - rain_thr, 0.0)

        # 计算API（可微）
        api = self.compute_api(rain_eff, api_decay)

        # 角度计算（年周期）
        ang1 = 2.0 * torch.pi * (self.doy / 365.0)
        ang2 = 4.0 * torch.pi * (self.doy / 365.0)

        # 基流分量（可微clip替代）
        base_log = b0 + s1 * torch.sin(ang1) + c1 * torch.cos(ang1) + s2 * torch.sin(ang2) + c2 * torch.cos(ang2)
        # 使用可微的饱和函数替代硬clip，保持梯度流
        base_log_clamped = torch.tanh(base_log / 20.0) * 20.0  # 近似clip到[-20,20]
        base = torch.exp(base_log_clamped)

        # 事件分量（可微）
        evt_lin = (
            p0
            + pr * torch.log1p(rain_eff)
            + pq * torch.log1p(torch.clamp_min(self.runoff, 0.0))
            + pa * torch.log1p(torch.clamp_min(api, 0.0))
            + pf * self.fert
        )
        evt = softplus_torch(evt_lin)

        # 组合分量
        raw = base + mix_kappa * evt
        raw = torch.clamp_min(raw, 1e-12)  # 确保正值，clamp_min在边界处梯度为0

        # 总量分配（守恒到year_total * year_scale）
        total = self.year_total * year_scale
        L = total * raw / (torch.sum(raw) + 1e-12)

        # -------- (B) learn f_surface（可微分约束）----------
        sf_b0, sf_r, sf_q, sf_a, sf_f, sf_sin, sf_cos = th[14], th[15], th[16], th[17], th[18], th[19], th[20]

        sf_lin = (
            sf_b0
            + sf_r * torch.log1p(rain_eff)
            + sf_q * torch.log1p(torch.clamp_min(self.runoff, 0.0))
            + sf_a * torch.log1p(torch.clamp_min(api, 0.0))
            + sf_f * self.fert
            + sf_sin * torch.sin(ang1)
            + sf_cos * torch.cos(ang1)
        )

        # 可微分范围约束：sigmoid映射到(0,1)，再线性缩放到[0.1, 0.98]
        f_surface_sigmoid = sigmoid_torch(sf_lin)
        f_surface = 0.1 + 0.88 * f_surface_sigmoid  # 确保在[0.1, 0.98]范围内

        # 调试信息（保持为torch tensor，不参与梯度计算）
        dbg = {
            'raw': raw.detach(),
            'base': base.detach(),
            'evt': evt.detach(),
            'api': api.detach(),
            'rain_eff': rain_eff.detach(),
            'f_surface': f_surface.detach()
        }

        return L, f_surface, dbg

    def reg_smooth(self, L: torch.Tensor) -> torch.Tensor:
        """
        日总量平滑正则项（二阶差分，可微）

        参数:
            L: 日总源项序列 (T,)

        返回:
            平滑正则损失（标量）
        """
        d1 = torch.diff(L)
        d2 = torch.diff(d1)
        return torch.mean(d2 ** 2)

    def reg_prior(self, L: torch.Tensor) -> torch.Tensor:
        """
        日总量先验正则项（与prior_indicator的MSE，可微）

        参数:
            L: 日总源项序列 (T,)

        返回:
            先验正则损失（标量），如果无先验则返回0
        """
        if self.prior is None:
            return torch.tensor(0.0, device=self.device)

        q = torch.clamp_min(L, 1e-12)
        q = q / (torch.sum(q) + 1e-12)

        return torch.mean((q - self.prior) ** 2)

    def reg_fs_smooth(self, fs: torch.Tensor) -> torch.Tensor:
        """
        地表比例平滑正则项（二阶差分，可微）

        参数:
            fs: 地表比例序列 (T,)

        返回:
            平滑正则损失（标量）
        """
        d1 = torch.diff(fs)
        d2 = torch.diff(d1)
        return torch.mean(d2 ** 2)

    def reg_fs_prior(self, fs: torch.Tensor) -> torch.Tensor:
        """
        地表比例先验正则项（与fs_prior的MSE，可微）

        参数:
            fs: 地表比例序列 (T,)

        返回:
            先验正则损失（标量），如果无先验则返回0
        """
        if self.fs_prior is None:
            return torch.tensor(0.0, device=self.device)

        return torch.mean((fs - self.fs_prior) ** 2)


# =============================================================================
# 3) torch版本空间分配函数
# =============================================================================

def build_source_maps_with_ml_torch(X, groups, idx_g, pix_r, pix_c, group_ratio,
                                     landuse_shape, L_daily, ml_w, ml_b, device='cpu'):
    """
    构建日尺度源项空间分布图（含ML权重学习，PyTorch版本）

    参数:
        X: 网格特征矩阵 (n_grid, n_feat)，可以是torch.Tensor或numpy数组
        groups: 网格所属土地利用组 (n_grid,)，可以是torch.Tensor或numpy数组
        idx_g: 字典 {group: 索引数组}，值为numpy数组或列表
        pix_r: 网格行坐标 (n_grid,)，可以是torch.Tensor或numpy数组
        pix_c: 网格列坐标 (n_grid,)，可以是torch.Tensor或numpy数组
        group_ratio: 字典 {group: 比例}，键为int，值为float
        landuse_shape: (rows, cols) 土地利用栅格形状，元组
        L_daily: 日总源项序列 (n_days,)，可以是torch.Tensor或numpy数组
        ml_w: ML权重向量 (n_feat,)，torch.Tensor，要求requires_grad=True以支持梯度
        ml_b: ML偏置标量，torch.Tensor，要求requires_grad=True以支持梯度
        device: 计算设备（'cpu'或'cuda'）

    返回:
        source_maps: 源项空间分布图列表，每个元素为 (rows, cols) torch.Tensor
    """
    # 转换为torch张量（保留梯度信息）
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if not isinstance(groups, torch.Tensor):
        groups = torch.as_tensor(groups, dtype=torch.long, device=device)
    if not isinstance(pix_r, torch.Tensor):
        pix_r = torch.as_tensor(pix_r, dtype=torch.long, device=device)
    if not isinstance(pix_c, torch.Tensor):
        pix_c = torch.as_tensor(pix_c, dtype=torch.long, device=device)
    if not isinstance(L_daily, torch.Tensor):
        L_daily = torch.as_tensor(L_daily, dtype=torch.float32, device=device)

    rows, cols = landuse_shape
    # 组内均匀分配：同一土地利用组内的像元具有相同源强，不再使用X、ml_w、ml_b
    rc_by_g = {}
    count_by_g = {}
    for g, inds in idx_g.items():
        if not isinstance(inds, torch.Tensor):
            inds = torch.as_tensor(inds, dtype=torch.long, device=device)
        rc_by_g[g] = (pix_r[inds], pix_c[inds])
        count_by_g[g] = max(int(inds.numel()), 1)

    # 组比例转换为tensor，方便后续计算
    group_ratio_tensor = {}
    for g in rc_by_g.keys():
        ratio = group_ratio.get(int(g), 0.0)
        group_ratio_tensor[g] = torch.tensor(ratio, dtype=torch.float32, device=device)

    source_maps = []
    for t in range(len(L_daily)):
        total = L_daily[t]  # 标量张量
        src = torch.zeros((rows, cols), dtype=torch.float32, device=device)

        for g in rc_by_g.keys():
            # 基于原始Python比例判断，避免调用.item()
            if group_ratio.get(int(g), 0.0) == 0.0:
                continue
            Lg = total * group_ratio_tensor[g]
            rr, cc = rc_by_g[g]
            src[rr, cc] = Lg / float(count_by_g[g])

        source_maps.append(src)

    return source_maps


# =============================================================================
# 4) torch版本物理过程函数
# =============================================================================

def par_allocation_torch(sink_classified, par_a, par_n, par_u):
    """
    torch版本参数按土地利用组分配（兼容NumPy输入）

    参数:
        sink_classified: 分类后的土地利用栅格 (rows, cols)，
                        值为1(耕地)、2(自然/草地)、3(不透水)
                        可以是torch.Tensor或numpy数组
        par_a: 耕地参数值（标量或与sink_classified相同形状的张量/数组）
        par_n: 自然/草地参数值
        par_u: 不透水参数值

    返回:
        p: 分配后的参数栅格 (rows, cols)，保持可微性，torch.Tensor类型
    """
    # 转换为torch张量，保留梯度信息
    if not isinstance(sink_classified, torch.Tensor):
        sink_classified = torch.as_tensor(sink_classified, dtype=torch.float32)
    if not isinstance(par_a, torch.Tensor):
        par_a = torch.as_tensor(par_a, dtype=torch.float32)
    if not isinstance(par_n, torch.Tensor):
        par_n = torch.as_tensor(par_n, dtype=torch.float32)
    if not isinstance(par_u, torch.Tensor):
        par_u = torch.as_tensor(par_u, dtype=torch.float32)

    # 确保参数值可以广播到sink_classified的形状
    # 如果参数是标量，广播会自动处理
    # 如果参数已经是与sink_classified相同形状的张量，则直接使用

    # 创建与sink_classified相同形状的零张量
    p = torch.zeros_like(sink_classified, dtype=torch.float32)

    # 使用掩码分配参数值，保持梯度流
    mask_a = (sink_classified == 1)
    mask_n = (sink_classified == 2)
    mask_u = (sink_classified == 3)

    # 将参数值广播到对应位置
    p = p + mask_a * par_a
    p = p + mask_n * par_n
    p = p + mask_u * par_u

    return p


def path_allocation_day_torch(pcp: torch.Tensor, surface_flow: torch.Tensor,
                              source: torch.Tensor, par: torch.Tensor,
                              f_surface: torch.Tensor, sink_classified: torch.Tensor) -> tuple:
    """
    torch版本日尺度路径分配

    关键改动：使用可学习的 f_surface(t) 切分 daily source -> surface vs legacy。
    注意：pcp 和 surface_flow 参数主要为保持接口兼容性而保留，在核心计算中未直接使用。

    参数:
        pcp: 降雨标量（保持兼容性）
        surface_flow: 地表流标量（保持兼容性）
        source: 源项空间分布图 (rows, cols)
        par: 物理参数向量 (至少51个元素)
        f_surface: 地表比例标量 (0~1)
        sink_classified: 分类后的土地利用栅格 (rows, cols)，值为1(耕地)、2(自然/草地)、3(不透水)

    返回:
        surf_dis: 地表溶解态 (rows, cols)
        surf_ads: 地表吸附态 (rows, cols)
        bgc_dis: 生物地球化学过程溶解态 (rows, cols)
        bgc_ads: 生物地球化学过程吸附态 (rows, cols)
        hyd_dis: 水文滞后过程溶解态 (rows, cols)
        hyd_ads: 水文滞后过程吸附态 (rows, cols)
    """
    # 参数分配（根据NumPy版本索引）
    source_allocation = par_allocation_torch(sink_classified, par[0], par[17], par[34])
    dis_leak1 = par_allocation_torch(sink_classified, par[1], par[18], par[35])
    dis_leak2 = par_allocation_torch(sink_classified, par[2], par[19], par[36])
    ads_leak1 = par_allocation_torch(sink_classified, par[3], par[20], par[37])
    ads_leak2 = par_allocation_torch(sink_classified, par[4], par[21], par[38])

    # 溶解态 vs 吸附态分配
    dissolved = source * source_allocation
    adsorb = source * (1 - source_allocation)

    # 使用可学习的地表比例切分
    frac = torch.clamp(f_surface, 0.0, 1.0)

    surf_dis = dissolved * frac
    surf_ads = adsorb * frac
    dis_leg = dissolved * (1 - frac)
    ads_leg = adsorb * (1 - frac)

    # 生物地球化学过程分配
    bgc_dis = dis_leg * (1 - dis_leak1)
    hyd_dis = dis_leg * dis_leak1 * (1 - dis_leak2)

    bgc_ads = ads_leg * (1 - ads_leak1)
    hyd_ads = ads_leg * ads_leak1 * (1 - ads_leak2)

    return surf_dis, surf_ads, bgc_dis, bgc_ads, hyd_dis, hyd_ads


def bgc_legacy_contribution_torch(new_nutrient, legacy_pool, par, form, sink_classified):
    """
    torch版本生物地球化学过程（溶解态/吸附态）

    参数:
        new_nutrient: 新增养分输入 (rows, cols) 或可广播形状，torch.Tensor
        legacy_pool: 遗留池 (rows, cols) 或可广播形状，torch.Tensor
        par: 物理参数向量 (至少51个元素)，torch.Tensor
        form: 形态，"dis"（溶解态）或"ads"（吸附态）
        sink_classified: 分类后的土地利用栅格 (rows, cols)，
                        值为1(耕地)、2(自然/草地)、3(不透水)，torch.Tensor

    返回:
        total: 当前步贡献总量 (rows, cols)
        new_pool: 更新后的遗留池 (rows, cols)
        bgc_to_hyd: 传递给水文滞后过程的量 (rows, cols)
    """
    if form == "dis":
        p_current = par_allocation_torch(sink_classified, par[5], par[22], par[39])
        p_history = par_allocation_torch(sink_classified, par[6], par[23], par[40])
        leak = par_allocation_torch(sink_classified, par[7], par[24], par[41])
    else:  # "ads"
        p_current = par_allocation_torch(sink_classified, par[8], par[25], par[42])
        p_history = par_allocation_torch(sink_classified, par[9], par[26], par[43])
        leak = par_allocation_torch(sink_classified, par[10], par[27], par[44])

    total = p_current * new_nutrient + p_history * legacy_pool
    raw_pool = legacy_pool + new_nutrient - total
    bgc_to_hyd = raw_pool * 0.1
    new_pool = raw_pool * (1 - 0.1) * (1 - leak)
    return total, new_pool, bgc_to_hyd


def hyd_legacy_step_torch(new_nutrient, legacy_pool, par, form, sink_classified):
    """
    torch版本水文滞后过程（溶解态/吸附态）

    参数:
        new_nutrient: 新增养分输入 (rows, cols) 或可广播形状，torch.Tensor
        legacy_pool: 遗留池 (rows, cols) 或可广播形状，torch.Tensor
        par: 物理参数向量 (至少51个元素)，torch.Tensor
        form: 形态，"dis"（溶解态）或"ads"（吸附态）
        sink_classified: 分类后的土地利用栅格 (rows, cols)，
                        值为1(耕地)、2(自然/草地)、3(不透水)，torch.Tensor

    返回:
        total: 当前步贡献总量 (rows, cols)
        new_pool: 更新后的遗留池 (rows, cols)
    """
    if form == "dis":
        p_history = par_allocation_torch(sink_classified, par[12], par[29], par[46])
        leak = par_allocation_torch(sink_classified, par[13], par[30], par[47])
    else:  # "ads"
        p_history = par_allocation_torch(sink_classified, par[15], par[32], par[49])
        leak = par_allocation_torch(sink_classified, par[16], par[33], par[50])

    total = legacy_pool * p_history
    new_pool = (legacy_pool + new_nutrient - total) * leak
    return total, new_pool


# =============================================================================
# torch版本增量衰减函数
# =============================================================================

def _attenuate_incremental_torch(val, sink, slope, i, j, par, mode, stats=None):
    """
    torch版本增量衰减（森林、草地、水体）

    参数:
        val: 输入值（torch.Tensor标量，requires_grad=True）
        sink: 土地利用栅格（numpy数组或torch.Tensor）
        slope: 坡度栅格（numpy数组或torch.Tensor）
        i, j: 行、列索引（整数）
        par: 物理参数向量（torch.Tensor，至少57个元素）
        mode: 形态，"dis"（溶解态）或"ads"（吸附态）
        stats: 可选字典，用于收集统计信息。如果提供，将更新以下键：
            'lu2_total': landuse == 2 的总次数
            'lu2_lt_threshold': landuse==2 且 val < threshold 的次数
            'lu2_else': landuse==2 且 val >= threshold 的次数
            'lu4_total': landuse == 4 的总次数
            'lu4_lt_threshold': landuse==4 且 val < threshold 的次数
            'lu4_else': landuse==4 且 val >= threshold 的次数
            'lu5_total': landuse == 5 的总次数
            'lu5_gt_threshold': landuse==5 且 val > threshold 的次数（注意：水体衰减条件是 val > threshold）
            'lu5_else': landuse==5 且 val <= threshold 的次数

    返回:
        val: 衰减后的值（torch.Tensor标量）
    """
    # 校验mode
    if mode not in ("dis", "ads"):
        raise ValueError(f"mode必须是'dis'或'ads'，收到: {mode}")

    # 确保sink和slope是torch张量（不要求梯度，仅用于索引）
    if not isinstance(sink, torch.Tensor):
        sink = torch.as_tensor(sink, dtype=torch.long, device=val.device)
    if not isinstance(slope, torch.Tensor):
        slope = torch.as_tensor(slope, dtype=torch.float32, device=val.device)

    lu = sink[i, j].item()  # 转换为Python整数

    # 统计landuse计数
    if stats is not None:
        if lu == 2:
            stats['lu2_total'] = stats.get('lu2_total', 0) + 1
        elif lu == 4:
            stats['lu4_total'] = stats.get('lu4_total', 0) + 1
        elif lu == 5:
            stats['lu5_total'] = stats.get('lu5_total', 0) + 1

    if lu == 2:  # FRST
        threshold = par[53]
        coeff_raw = (5.889 + 0.1609 * slope[i, j] - 0.0353 + 1.007 - 0.4511 * 0.4 + 59.8298) / 100.0
        coeff = torch.clamp(coeff_raw + 0.20, min=0.75, max=0.95)
        # 统计分支
        if stats is not None:
            if val.item() < threshold.item():
                stats['lu2_lt_threshold'] = stats.get('lu2_lt_threshold', 0) + 1
            else:
                stats['lu2_else'] = stats.get('lu2_else', 0) + 1
        # 使用torch.where保持梯度流
        val = torch.where(val < threshold,
                          val * coeff,
                          val - threshold)  # mode已校验，直接减法
    elif lu == 4:  # PAST
        threshold = par[55]
        coeff = (5.889 + 0.1609 * slope[i, j] - 0.0353 + 1.007 - 0.4511 * 0.13 + 59.8298) / 100.0
        # 统计分支
        if stats is not None:
            if val.item() < threshold.item():
                stats['lu4_lt_threshold'] = stats.get('lu4_lt_threshold', 0) + 1
            else:
                stats['lu4_else'] = stats.get('lu4_else', 0) + 1
        val = torch.where(val < threshold,
                          val * coeff,
                          val - threshold)
    elif lu == 5:  # WATR
        threshold = par[56]
        # 注意：vv = max(val, 1e-6)，使用clamp_min
        vv = torch.clamp_min(val, 1e-6)
        coeff = (0.0797 * torch.exp(-0.00518 * (2700.0 / vv)) + 65.5432) / 100.0
        # 统计分支
        if stats is not None:
            if val.item() > threshold.item():
                stats['lu5_gt_threshold'] = stats.get('lu5_gt_threshold', 0) + 1
            else:
                stats['lu5_else'] = stats.get('lu5_else', 0) + 1
        # 条件：val > threshold
        val = torch.where(val > threshold,
                          val * coeff,
                          val * 0.0)  # 等价于置零，但保留梯度
    # 若lu不是2、4、5，则直接返回原值
    return val


# =============================================================================
# torch版本逆向追踪函数
# =============================================================================

def trace_back_iterative_torch(total_source_map, dir_up, postorder_nodes, xy, sink, slope, par, mode, stats=None):
    """
    torch版本逆向追踪计算（从上游到出口）

    参数:
        total_source_map: 总源项图 (rows, cols)，torch.Tensor，requires_grad=True
        dir_up: 流向拓扑字典 {cell_id: [upstream_cells]}
        postorder_nodes: 后序遍历节点列表
        xy: 节点坐标字典 {cell_id: (row, col)}
        sink: 土地利用栅格 (rows, cols)，numpy数组或torch.Tensor
        slope: 坡度栅格 (rows, cols)，numpy数组或torch.Tensor
        par: 物理参数向量 (至少57个元素)，torch.Tensor
        mode: 形态，"dis"（溶解态）或"ads"（吸附态）
        stats: 可选字典，用于收集衰减统计信息，传递给 _attenuate_incremental_torch

    返回:
        out: 逆向追踪后的整张图 (rows, cols)，torch.Tensor
    """
    if mode not in ("dis", "ads"):
        raise ValueError(f"mode必须是'dis'或'ads'，收到: {mode}")

    # 创建输出张量，保留梯度流
    out = total_source_map.clone()

    # 按后序遍历节点顺序处理
    for node in postorder_nodes:
        ups = dir_up.get(node, None)
        if not ups:
            continue
        i, j = xy[node]

        # 先累加所有上游贡献
        total_upstream = torch.tensor(0.0, device=out.device)
        for up in ups:
            if up not in xy:
                continue
            ui, uj = xy[up]
            total_upstream = total_upstream + out[ui, uj]

        # 累加当前节点值，然后衰减一次
        updated_val = out[i, j] + total_upstream
        updated_val = _attenuate_incremental_torch(updated_val, sink, slope, i, j, par, mode, stats)
        out[i, j] = updated_val

    # 返回整张图，不取出口值
    return out


# =============================================================================
# torch版本局部物理最小链路函数
# =============================================================================

def run_local_physics_step_torch(source, par, f_surface, sink_classified,
                                 bgc_dis_pool, bgc_ads_pool,
                                 hyd_dis_pool, hyd_ads_pool,
                                 pcp, surface_flow):
    """
    torch版本单日、单步局部物理最小链路

    串联已完成的三个torch局部模块：
    1. path_allocation_day_torch
    2. bgc_legacy_contribution_torch
    3. hyd_legacy_step_torch

    参数:
        source: 单日源项空间图 (rows, cols)，torch.Tensor
        par: 物理参数向量 (至少51个元素)，torch.Tensor，requires_grad=True
        f_surface: 地表比例标量 (0~1)，torch.Tensor（可带梯度）
        sink_classified: 分类后的土地利用栅格 (rows, cols)，
                        值为1(耕地)、2(自然/草地)、3(不透水)，torch.Tensor
        bgc_dis_pool: 初始溶解态bgc池 (rows, cols)，torch.Tensor
        bgc_ads_pool: 初始吸附态bgc池 (rows, cols)，torch.Tensor
        hyd_dis_pool: 初始溶解态hyd池 (rows, cols)，torch.Tensor
        hyd_ads_pool: 初始吸附态hyd池 (rows, cols)，torch.Tensor
        pcp: 降雨标量（保持接口兼容），torch.Tensor
        surface_flow: 地表流标量（保持接口兼容），torch.Tensor

    返回:
        local_output: 局部输出标量（总和），torch.Tensor
        surf_dis: 地表溶解态 (rows, cols)
        surf_ads: 地表吸附态 (rows, cols)
        bgc_dis_total: bgc溶解态贡献 (rows, cols)
        bgc_ads_total: bgc吸附态贡献 (rows, cols)
        hyd_dis_total: hyd溶解态贡献 (rows, cols)
        hyd_ads_total: hyd吸附态贡献 (rows, cols)
        bgc_dis_pool_new: 更新后溶解态bgc池 (rows, cols)
        bgc_ads_pool_new: 更新后吸附态bgc池 (rows, cols)
        hyd_dis_pool_new: 更新后溶解态hyd池 (rows, cols)
        hyd_ads_pool_new: 更新后吸附态hyd池 (rows, cols)
    """
    # 1. 调用 path_allocation_day_torch
    surf_dis, surf_ads, bgc_dis, bgc_ads, hyd_dis, hyd_ads = path_allocation_day_torch(
        pcp, surface_flow, source, par, f_surface, sink_classified
    )

    # 2. 调用 bgc_legacy_contribution_torch("dis")
    bgc_dis_total, bgc_dis_pool_new, bgc2hyd_dis = bgc_legacy_contribution_torch(
        bgc_dis, bgc_dis_pool, par, "dis", sink_classified
    )

    # 3. 调用 bgc_legacy_contribution_torch("ads")
    bgc_ads_total, bgc_ads_pool_new, bgc2hyd_ads = bgc_legacy_contribution_torch(
        bgc_ads, bgc_ads_pool, par, "ads", sink_classified
    )

    # 4. 调用 hyd_legacy_step_torch("dis")
    hyd_dis_total, hyd_dis_pool_new = hyd_legacy_step_torch(
        hyd_dis + bgc2hyd_dis, hyd_dis_pool, par, "dis", sink_classified
    )

    # 5. 调用 hyd_legacy_step_torch("ads")
    hyd_ads_total, hyd_ads_pool_new = hyd_legacy_step_torch(
        hyd_ads + bgc2hyd_ads, hyd_ads_pool, par, "ads", sink_classified
    )

    # 6. 构造最小局部输出（不引入拓扑传播）
    local_output = (
        surf_dis.sum() + surf_ads.sum()
        + bgc_dis_total.sum() + bgc_ads_total.sum()
        + hyd_dis_total.sum() + hyd_ads_total.sum()
    )

    # 7. 返回结果
    return (local_output,
            surf_dis, surf_ads,
            bgc_dis_total, bgc_ads_total,
            hyd_dis_total, hyd_ads_total,
            bgc_dis_pool_new, bgc_ads_pool_new,
            hyd_dis_pool_new, hyd_ads_pool_new)


# =============================================================================
# torch版本单日含拓扑传播训练链路
# =============================================================================

def run_single_day_with_trace_torch(source, par, f_surface, sink_classified,
                                    sink, slope, dir_up, postorder_nodes, xy,
                                    bgc_dis_pool, bgc_ads_pool,
                                    hyd_dis_pool, hyd_ads_pool,
                                    pcp, surface_flow, outlet_code, stats=None):
    """
    torch版本单日含拓扑传播训练链路

    串联局部物理过程和空间传播：
    1. run_local_physics_step_torch
    2. trace_back_iterative_torch（溶解态/吸附态）
    3. 取出口值，构造单日总输出

    参数:
        source: 单日源项空间图 (rows, cols)，torch.Tensor
        par: 物理参数向量，torch.Tensor，requires_grad=True
        f_surface: 单个标量，torch.Tensor
        sink_classified: 分类后的土地利用图 (rows, cols)，torch.Tensor
        sink: 原始 landuse 图 (rows, cols)，numpy 或 torch
        slope: 坡度图 (rows, cols)，numpy 或 torch
        dir_up: 流向拓扑字典
        postorder_nodes: 后序遍历节点列表
        xy: 节点坐标字典
        bgc_dis_pool: 初始溶解态 bgc 池
        bgc_ads_pool: 初始吸附态 bgc 池
        hyd_dis_pool: 初始溶解态 hyd 池
        hyd_ads_pool: 初始吸附态 hyd 池
        pcp: 单日降雨标量
        surface_flow: 单日地表流标量
        outlet_code: 出口编码字符串，例如 cfg["OUTLET_CODE"]
        stats: 可选字典，用于收集衰减统计信息，传递给 trace_back_iterative_torch

    返回:
        day_output: 单日总输出（标量），torch.Tensor
        out_dis: 出口溶解态值（标量），torch.Tensor
        out_ads: 出口吸附态值（标量），torch.Tensor
        hyd_out: hyd部分输出（标量），torch.Tensor
        out_dis_map: 传播后溶解态图 (rows, cols)，torch.Tensor
        out_ads_map: 传播后吸附态图 (rows, cols)，torch.Tensor
        bgc_dis_pool_new: 更新后溶解态bgc池
        bgc_ads_pool_new: 更新后吸附态bgc池
        hyd_dis_pool_new: 更新后溶解态hyd池
        hyd_ads_pool_new: 更新后吸附态hyd池
    """

    # 1. 调用现有 run_local_physics_step_torch
    (local_output,
     surf_dis, surf_ads,
     bgc_dis_total, bgc_ads_total,
     hyd_dis_total, hyd_ads_total,
     bgc_dis_pool_new, bgc_ads_pool_new,
     hyd_dis_pool_new, hyd_ads_pool_new) = run_local_physics_step_torch(
        source, par, f_surface, sink_classified,
        bgc_dis_pool, bgc_ads_pool,
        hyd_dis_pool, hyd_ads_pool,
        pcp, surface_flow
    )

    # 2. 构造传播输入
    total_dis_source = surf_dis + bgc_dis_total
    total_ads_source = surf_ads + bgc_ads_total

    # 3. 分别调用 trace_back_iterative_torch
    out_dis_map = trace_back_iterative_torch(
        total_dis_source, dir_up, postorder_nodes, xy,
        sink, slope, par, mode="dis", stats=stats
    )
    out_ads_map = trace_back_iterative_torch(
        total_ads_source, dir_up, postorder_nodes, xy,
        sink, slope, par, mode="ads", stats=stats
    )

    # 4. 用 outlet_code 取出口值
    oi, oj = int(outlet_code[:4]), int(outlet_code[4:])
    out_dis = out_dis_map[oi, oj]
    out_ads = out_ads_map[oi, oj]

    # 5. hyd 部分也做空间传播，使出口总量和空间图口径一致
    hyd_dis_map = trace_back_iterative_torch(
        hyd_dis_total, dir_up, postorder_nodes, xy,
        sink, slope, par, mode="dis", stats=stats
    )
    hyd_ads_map = trace_back_iterative_torch(
        hyd_ads_total, dir_up, postorder_nodes, xy,
        sink, slope, par, mode="ads", stats=stats
    )
    hyd_out = hyd_dis_map[oi, oj] + hyd_ads_map[oi, oj]

    # 6. 构造单日总输出
    day_output = out_dis + out_ads + hyd_out

    # 7. 返回结果
    return (day_output, out_dis, out_ads, hyd_out,
            out_dis_map, out_ads_map,
            bgc_dis_pool_new, bgc_ads_pool_new,
            hyd_dis_pool_new, hyd_ads_pool_new)


# =============================================================================
# 5) torch版本三水库三通道水文核
# =============================================================================

def slice_three_store_hydrology_params_torch(par, sink_classified):
    """
    从参数向量中切片三水库水文核参数并按土地利用分配

    假设现有par长度为N，在尾部追加11个参数：
    N+0  k_surf_a
    N+1  k_surf_n
    N+2  k_surf_u
    N+3  k_soil_a
    N+4  k_soil_n
    N+5  k_soil_u
    N+6  k_gw_a
    N+7  k_gw_n
    N+8  k_gw_u
    N+9  alpha_surf_to_soil
    N+10 alpha_soil_to_gw

    前9个参数按土地利用分配到二维栅格；后2个是全局标量。
    优先复用现有 par_allocation_torch。

    参数:
        par: 参数向量 (至少11个元素)，torch.Tensor
        sink_classified: 分类土地利用栅格 (rows, cols)，值为1(耕地)、2(自然/草地)、3(不透水)

    返回:
        k_surf_grid: 地表水库出流系数栅格 (rows, cols)
        k_soil_grid: 土壤水库出流系数栅格 (rows, cols)
        k_gw_grid: 地下水水库出流系数栅格 (rows, cols)
        alpha_surf_to_soil: 地表到土壤传递系数 (标量)
        alpha_soil_to_gw: 土壤到地下水传递系数 (标量)
    """
    # 确保输入为torch张量
    if not isinstance(par, torch.Tensor):
        par = torch.as_tensor(par, dtype=torch.float32, device=sink_classified.device if isinstance(sink_classified, torch.Tensor) else 'cpu')
    if not isinstance(sink_classified, torch.Tensor):
        sink_classified = torch.as_tensor(sink_classified, dtype=torch.float32, device=par.device)

    # 检查参数长度是否足够
    if par.numel() < 11:
        raise ValueError(f"par长度至少需要11个元素，当前为{par.numel()}")

    # 动态计算偏移量：假设新参数追加在尾部
    offset = par.numel() - 11  # 现有参数长度N

    # 切片参数（索引offset到offset+10）
    k_surf_a = par[offset]
    k_surf_n = par[offset + 1]
    k_surf_u = par[offset + 2]
    k_soil_a = par[offset + 3]
    k_soil_n = par[offset + 4]
    k_soil_u = par[offset + 5]
    k_gw_a = par[offset + 6]
    k_gw_n = par[offset + 7]
    k_gw_u = par[offset + 8]
    alpha_surf_to_soil = par[offset + 9]
    alpha_soil_to_gw = par[offset + 10]

    # 使用 par_allocation_torch 分配土地利用相关参数
    k_surf_grid = par_allocation_torch(sink_classified, k_surf_a, k_surf_n, k_surf_u)
    k_soil_grid = par_allocation_torch(sink_classified, k_soil_a, k_soil_n, k_soil_u)
    k_gw_grid = par_allocation_torch(sink_classified, k_gw_a, k_gw_n, k_gw_u)

    return k_surf_grid, k_soil_grid, k_gw_grid, alpha_surf_to_soil, alpha_soil_to_gw


def three_store_hydrology_step_torch(P, S_surf_prev, S_soil_prev, S_gw_prev,
                                     k_surf_grid, k_soil_grid, k_gw_grid,
                                     alpha_surf_to_soil, alpha_soil_to_gw,
                                     return_transfers=False, return_diagnostics=False):
    """
    三水库三通道水文核单步更新（可微）

    使用简洁稳定、保持可微的版本：
    - 固定非线性指数：BETA_SURF=0.9, BETA_SOIL=1.0, BETA_GW=1.1
    - 需要显式加入：非负约束、库存约束、剩余库存基础上的库间传递
    - 不能让出流超过当前库存
    - 不能让状态变成负数

    参数:
        P: 单日降雨 (标量) 或已广播到二维栅格 (rows, cols)
        S_surf_prev: 上一时刻地表水库库存 (rows, cols)
        S_soil_prev: 上一时刻土壤水库库存 (rows, cols)
        S_gw_prev: 上一时刻地下水水库库存 (rows, cols)
        k_surf_grid: 地表水库出流系数栅格 (rows, cols)
        k_soil_grid: 土壤水库出流系数栅格 (rows, cols)
        k_gw_grid: 地下水水库出流系数栅格 (rows, cols)
        alpha_surf_to_soil: 地表到土壤传递系数 (标量)
        alpha_soil_to_gw: 土壤到地下水传递系数 (标量)
        return_transfers: 是否返回库间传递量 to_soil, to_gw (默认False)

    返回:
        S_surf_new: 新地表水库库存 (rows, cols)
        S_soil_new: 新土壤水库库存 (rows, cols)
        S_gw_new: 新地下水水库库存 (rows, cols)
        Q_surf: 地表出流 (rows, cols)
        Q_inter: 壤中流出流 (rows, cols)
        Q_base: 基流出流 (rows, cols)
        to_soil: 地表到土壤传递量 (rows, cols)，仅当 return_transfers=True 时返回
        to_gw: 土壤到地下水传递量 (rows, cols)，仅当 return_transfers=True 时返回
    """
    # 固定非线性指数
    BETA_SURF = 0.9
    BETA_SOIL = 1.0
    BETA_GW = 1.1
    # 数值稳定处理：避免幂运算在接近零时梯度爆炸
    eps = 1e-8

    # 确保P是二维栅格（如果P是标量，则广播到S_surf_prev的形状）
    if isinstance(P, (int, float)) or (isinstance(P, torch.Tensor) and P.ndim == 0):
        P = torch.full_like(S_surf_prev, float(P))

    # 1. 地表水库更新
    # 当天可用水量 = 上一时刻库存 + 降雨
    available_surf = S_surf_prev + P
    # 数值稳定处理：避免幂运算在接近零时梯度爆炸
    available_surf_stable = torch.clamp_min(available_surf, eps)
    # 非线性出流计算，限制出流不超过可用水量
    Q_surf_raw = k_surf_grid * (available_surf_stable ** BETA_SURF)
    Q_surf = torch.min(Q_surf_raw, available_surf)  # 不能超过可用水量
    Q_surf = torch.clamp_min(Q_surf, 0.0)  # 非负约束

    # 出流后剩余库存 = 可用水量 - 出流
    S_surf_after_outflow = available_surf - Q_surf
    S_surf_after_outflow = torch.clamp_min(S_surf_after_outflow, 0.0)

    # 地表到土壤传递（基于出流后剩余库存中超过保护量的部分）
    protect_surf = 1.0  # mm，地表库保护量
    protect_soil = 1.0  # mm，土壤库保护量
    excess_surf = torch.clamp_min(S_surf_after_outflow - protect_surf, 0.0)
    to_soil = alpha_surf_to_soil * excess_surf
    to_soil = torch.min(to_soil, S_surf_after_outflow)  # 不能超过剩余库存
    to_soil = torch.clamp_min(to_soil, 0.0)

    # 地表最终库存 = 出流后剩余库存 - 传递给土壤的部分
    S_surf_new = S_surf_after_outflow - to_soil
    S_surf_new = torch.clamp_min(S_surf_new, 0.0)

    # 2. 土壤水库更新
    # 当天可用水量 = 上一时刻库存 + 来自地表的传递
    available_soil = S_soil_prev + to_soil
    # 数值稳定处理
    available_soil_stable = torch.clamp_min(available_soil, eps)
    # 非线性出流计算（壤中流），限制出流不超过可用水量
    Q_inter_raw = k_soil_grid * (available_soil_stable ** BETA_SOIL)
    Q_inter = torch.min(Q_inter_raw, available_soil)  # 不能超过可用水量
    Q_inter = torch.clamp_min(Q_inter, 0.0)

    # 出流后剩余库存 = 可用水量 - 出流
    S_soil_after_outflow = available_soil - Q_inter
    S_soil_after_outflow = torch.clamp_min(S_soil_after_outflow, 0.0)

    # 土壤到地下水传递（基于出流后剩余库存中超过保护量的部分）
    excess_soil = torch.clamp_min(S_soil_after_outflow - protect_soil, 0.0)
    to_gw = alpha_soil_to_gw * excess_soil
    to_gw = torch.min(to_gw, S_soil_after_outflow)  # 不能超过剩余库存
    to_gw = torch.clamp_min(to_gw, 0.0)

    # 土壤最终库存 = 出流后剩余库存 - 传递给地下水的部分
    S_soil_new = S_soil_after_outflow - to_gw
    S_soil_new = torch.clamp_min(S_soil_new, 0.0)

    # 3. 地下水水库更新
    # 当天可用水量 = 上一时刻库存 + 来自土壤的传递
    available_gw = S_gw_prev + to_gw
    # 数值稳定处理
    available_gw_stable = torch.clamp_min(available_gw, eps)
    # 非线性出流计算（基流），限制出流不超过可用水量
    Q_base_raw = k_gw_grid * (available_gw_stable ** BETA_GW)
    Q_base = torch.min(Q_base_raw, available_gw)  # 不能超过可用水量
    Q_base = torch.clamp_min(Q_base, 0.0)

    # 地下水最终库存 = 可用水量 - 出流
    S_gw_new = available_gw - Q_base
    S_gw_new = torch.clamp_min(S_gw_new, 0.0)

    # 诊断输出
    if return_diagnostics:
        # 计算每日总和
        Q_surf_sum = Q_surf.sum()
        Q_inter_sum = Q_inter.sum()
        Q_base_sum = Q_base.sum()
        to_soil_sum = to_soil.sum()
        to_gw_sum = to_gw.sum()
        S_surf_sum = S_surf_new.sum()
        S_soil_sum = S_soil_new.sum()
        S_gw_sum = S_gw_new.sum()

        diagnostics = {
            'Q_surf_sum': Q_surf_sum,
            'Q_inter_sum': Q_inter_sum,
            'Q_base_sum': Q_base_sum,
            'to_soil_sum': to_soil_sum,
            'to_gw_sum': to_gw_sum,
            'S_surf_sum': S_surf_sum,
            'S_soil_sum': S_soil_sum,
            'S_gw_sum': S_gw_sum
        }

        if return_transfers:
            return S_surf_new, S_soil_new, S_gw_new, Q_surf, Q_inter, Q_base, to_soil, to_gw, diagnostics
        else:
            return S_surf_new, S_soil_new, S_gw_new, Q_surf, Q_inter, Q_base, diagnostics
    elif return_transfers:
        return S_surf_new, S_soil_new, S_gw_new, Q_surf, Q_inter, Q_base, to_soil, to_gw
    else:
        return S_surf_new, S_soil_new, S_gw_new, Q_surf, Q_inter, Q_base


def simulate_three_store_hydrology_torch(rain, par, sink_classified, init_states=None, return_diagnostics=False):
    """
    模拟多日三水库水文过程

    参数:
        rain: 降雨序列 (T,)，1D torch.Tensor
        par: 参数向量 (至少11个元素)，torch.Tensor
        sink_classified: 分类土地利用栅格 (rows, cols)，值为1(耕地)、2(自然/草地)、3(不透水)
        init_states: 可选初始状态字典，包含键:
            'S_surf': 初始地表库存 (rows, cols)
            'S_soil': 初始土壤库存 (rows, cols)
            'S_gw': 初始地下水库存 (rows, cols)
            如果为None，则初始化为零
        return_diagnostics: 是否返回每日诊断信息

    返回:
        S_surf_seq: 地表库存序列 (T, rows, cols)
        S_soil_seq: 土壤库存序列 (T, rows, cols)
        S_gw_seq: 地下水库存序列 (T, rows, cols)
        Q_surf_seq: 地表出流序列 (T, rows, cols)
        Q_inter_seq: 壤中流出流序列 (T, rows, cols)
        Q_base_seq: 基流出流序列 (T, rows, cols)
        如果 return_diagnostics=True，则额外返回 diagnostics_seq: 每日诊断字典列表
    """
    # 确保输入为torch张量
    if not isinstance(rain, torch.Tensor):
        rain = torch.as_tensor(rain, dtype=torch.float32)
    if not isinstance(par, torch.Tensor):
        par = torch.as_tensor(par, dtype=torch.float32)
    if not isinstance(sink_classified, torch.Tensor):
        sink_classified = torch.as_tensor(sink_classified, dtype=torch.float32)

    device = rain.device
    par = par.to(device)
    sink_classified = sink_classified.to(device)

    # 获取栅格形状
    rows, cols = sink_classified.shape

    # 切片参数
    k_surf_grid, k_soil_grid, k_gw_grid, alpha_surf_to_soil, alpha_soil_to_gw = \
        slice_three_store_hydrology_params_torch(par, sink_classified)

    # 初始化状态
    if init_states is None:
        S_surf = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        S_soil = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        S_gw = torch.zeros((rows, cols), dtype=torch.float32, device=device)
    else:
        S_surf = init_states['S_surf'].to(device)
        S_soil = init_states['S_soil'].to(device)
        S_gw = init_states['S_gw'].to(device)

    # 准备输出序列
    T = len(rain)
    S_surf_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    S_soil_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    S_gw_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    Q_surf_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    Q_inter_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    Q_base_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)

    # 诊断序列（如果需要）
    diagnostics_seq = [] if return_diagnostics else None

    # 逐日模拟
    for t in range(T):
        P = rain[t]  # 标量

        # 调用单步更新
        if return_diagnostics:
            S_surf_new, S_soil_new, S_gw_new, Q_surf, Q_inter, Q_base, diagnostics = \
                three_store_hydrology_step_torch(
                    P, S_surf, S_soil, S_gw,
                    k_surf_grid, k_soil_grid, k_gw_grid,
                    alpha_surf_to_soil, alpha_soil_to_gw,
                    return_transfers=False,
                    return_diagnostics=True
                )
            diagnostics_seq.append(diagnostics)
        else:
            S_surf_new, S_soil_new, S_gw_new, Q_surf, Q_inter, Q_base = \
                three_store_hydrology_step_torch(
                    P, S_surf, S_soil, S_gw,
                    k_surf_grid, k_soil_grid, k_gw_grid,
                    alpha_surf_to_soil, alpha_soil_to_gw
                )

        # 存储结果
        S_surf_seq[t] = S_surf_new
        S_soil_seq[t] = S_soil_new
        S_gw_seq[t] = S_gw_new
        Q_surf_seq[t] = Q_surf
        Q_inter_seq[t] = Q_inter
        Q_base_seq[t] = Q_base

        # 更新状态
        S_surf = S_surf_new
        S_soil = S_soil_new
        S_gw = S_gw_new

    if return_diagnostics:
        return S_surf_seq, S_soil_seq, S_gw_seq, Q_surf_seq, Q_inter_seq, Q_base_seq, diagnostics_seq
    else:
        return S_surf_seq, S_soil_seq, S_gw_seq, Q_surf_seq, Q_inter_seq, Q_base_seq


def validate_three_store_hydrology_results(rain, S_surf_seq, S_soil_seq, S_gw_seq,
                                           Q_surf_seq, Q_inter_seq, Q_base_seq,
                                           init_states=None, tolerance=1e-6):
    """
    验证三水库水文结果

    检查：
    1. 状态非负
    2. 无NaN/Inf
    3. 质量守恒（降雨输入 ≈ 出流总和 + 库存变化）

    参数:
        rain: 降雨序列 (T,) 或 (T, rows, cols)，torch.Tensor
        S_surf_seq: 地表库存序列 (T, rows, cols)
        S_soil_seq: 土壤库存序列 (T, rows, cols)
        S_gw_seq: 地下水库存序列 (T, rows, cols)
        Q_surf_seq: 地表出流序列 (T, rows, cols)
        Q_inter_seq: 壤中流出流序列 (T, rows, cols)
        Q_base_seq: 基流出流序列 (T, rows, cols)
        init_states: 初始状态字典，可选，包含键 'S_surf', 'S_soil', 'S_gw'，每个为 (rows, cols) 张量。
                    如果为 None，则假设初始状态为零（模拟开始时库存为零）。
        tolerance: 质量守恒允许的误差容限

    返回:
        validation_dict: 验证结果字典，包含各项检查结果
    """
    # 1. 非负检查
    nonneg_surf = torch.all(S_surf_seq >= 0)
    nonneg_soil = torch.all(S_soil_seq >= 0)
    nonneg_gw = torch.all(S_gw_seq >= 0)
    nonneg_qsurf = torch.all(Q_surf_seq >= 0)
    nonneg_qinter = torch.all(Q_inter_seq >= 0)
    nonneg_qbase = torch.all(Q_base_seq >= 0)

    # 2. NaN/Inf检查
    def check_finite(tensor):
        return torch.all(torch.isfinite(tensor))

    finite_surf = check_finite(S_surf_seq)
    finite_soil = check_finite(S_soil_seq)
    finite_gw = check_finite(S_gw_seq)
    finite_qsurf = check_finite(Q_surf_seq)
    finite_qinter = check_finite(Q_inter_seq)
    finite_qbase = check_finite(Q_base_seq)

    # 3. 质量守恒检查
    # 确保rain是合适的形状以便求和
    if rain.ndim == 1:
        # (T,) 广播到每个栅格单元
        total_rain_input = rain.sum() * S_surf_seq[0].numel()
    elif rain.ndim == 3:
        # (T, rows, cols)
        total_rain_input = rain.sum()
    else:
        raise ValueError(f"rain维度应为1或3，实际为{rain.ndim}")

    # 计算总出流
    total_Q = Q_surf_seq.sum() + Q_inter_seq.sum() + Q_base_seq.sum()

    # 计算总库存变化
    if init_states is not None:
        S_total_start = (init_states['S_surf'].sum() +
                         init_states['S_soil'].sum() +
                         init_states['S_gw'].sum())
    else:
        # 假设模拟从零库存开始
        S_total_start = torch.tensor(0.0, device=S_surf_seq.device)

    S_total_end = S_surf_seq[-1].sum() + S_soil_seq[-1].sum() + S_gw_seq[-1].sum()
    delta_S = S_total_end - S_total_start


    # 质量平衡误差
    mass_balance_error = total_rain_input - (total_Q + delta_S)
    mass_balance_abs_error = torch.abs(mass_balance_error)
    mass_conservation = mass_balance_abs_error <= tolerance

    validation_dict = {
        'nonnegative': {
            'S_surf': nonneg_surf.item() if nonneg_surf.numel() == 1 else nonneg_surf,
            'S_soil': nonneg_soil.item() if nonneg_soil.numel() == 1 else nonneg_soil,
            'S_gw': nonneg_gw.item() if nonneg_gw.numel() == 1 else nonneg_gw,
            'Q_surf': nonneg_qsurf.item() if nonneg_qsurf.numel() == 1 else nonneg_qsurf,
            'Q_inter': nonneg_qinter.item() if nonneg_qinter.numel() == 1 else nonneg_qinter,
            'Q_base': nonneg_qbase.item() if nonneg_qbase.numel() == 1 else nonneg_qbase,
        },
        'finite': {
            'S_surf': finite_surf.item() if finite_surf.numel() == 1 else finite_surf,
            'S_soil': finite_soil.item() if finite_soil.numel() == 1 else finite_soil,
            'S_gw': finite_gw.item() if finite_gw.numel() == 1 else finite_gw,
            'Q_surf': finite_qsurf.item() if finite_qsurf.numel() == 1 else finite_qsurf,
            'Q_inter': finite_qinter.item() if finite_qinter.numel() == 1 else finite_qinter,
            'Q_base': finite_qbase.item() if finite_qbase.numel() == 1 else finite_qbase,
        },
        'mass_conservation': mass_conservation.item() if isinstance(mass_conservation, torch.Tensor) else mass_conservation,
        'mass_balance_details': {
            'total_rain_input': total_rain_input.item(),
            'total_outflow': total_Q.item(),
            'storage_change': delta_S.item(),
            'mass_balance_error': mass_balance_error.item(),
            'mass_balance_abs_error': mass_balance_abs_error.item(),
        }
    }

    return validation_dict


# =============================================================================
# 6) torch版本三水库三通道氮库核
# =============================================================================

def slice_three_store_nitrogen_params_torch(par, sink_classified):
    """
    从参数向量中切片三水库氮库核参数

    假设现有par长度为N（原始参数），第一阶段水文参数追加11个，
    第二阶段氮库参数追加5个在最后：
    N+11 c_surf_release
    N+12 c_soil_release
    N+13 c_gw_release
    N+14 beta_surf_to_soil
    N+15 beta_soil_to_gw

    所有5个参数均为全局标量，不按土地利用分配。

    参数:
        par: 参数向量 (至少16个元素)，torch.Tensor
        sink_classified: 分类土地利用栅格 (rows, cols)，仅用于接口兼容性

    返回:
        c_surf_release: 地表氮库释放系数（全局标量）
        c_soil_release: 土壤氮库释放系数（全局标量）
        c_gw_release: 地下氮库释放系数（全局标量）
        beta_surf_to_soil: 表层→土壤氮传递系数（全局标量）
        beta_soil_to_gw: 土壤→地下氮传递系数（全局标量）
    """
    # 确保输入为torch张量
    if not isinstance(par, torch.Tensor):
        par = torch.as_tensor(par, dtype=torch.float32, device=sink_classified.device if isinstance(sink_classified, torch.Tensor) else 'cpu')

    # 检查参数长度是否足够（至少16个：现有N + 水文11 + 氮库5）
    if par.numel() < 16:
        raise ValueError(f"par长度至少需要16个元素以包含氮库参数，当前为{par.numel()}")

    # 动态计算偏移量：氮库参数追加在水文参数之后
    offset = par.numel() - 16  # 现有参数长度N
    nitrogen_offset = offset + 11  # 跳过水文11个参数

    # 切片氮库参数
    c_surf_release = par[nitrogen_offset]
    c_soil_release = par[nitrogen_offset + 1]
    c_gw_release = par[nitrogen_offset + 2]
    beta_surf_to_soil = par[nitrogen_offset + 3]
    beta_soil_to_gw = par[nitrogen_offset + 4]

    return c_surf_release, c_soil_release, c_gw_release, beta_surf_to_soil, beta_soil_to_gw


def three_store_nitrogen_step_torch(source_day, N_surf_prev, N_soil_prev, N_gw_prev,
                                    Q_surf, Q_inter, Q_base, par, sink_classified):
    """
    三水库三通道氮库核单步更新（可微）

    氮释放公式：L = N_available * (1 - exp(-c * Q))
    其中：
      N_available: 当前氮库可用库存
      c: 对应释放系数 (c_surf_release, c_soil_release, c_gw_release)
      Q: 对应水文通道流量 (Q_surf, Q_inter, Q_base)

    关键约束：
    1. 释放量不超过当前可用库存
    2. 传递量不超过释放后剩余库存
    3. 所有状态和输出非负

    参数:
        source_day: 单日氮输入图 (rows, cols)
        N_surf_prev: 上一时刻地表氮库存 (rows, cols)
        N_soil_prev: 上一时刻土壤氮库存 (rows, cols)
        N_gw_prev: 上一时刻地下氮库存 (rows, cols)
        Q_surf: 单日地表出流图 (rows, cols)
        Q_inter: 单日壤中流出流图 (rows, cols)
        Q_base: 单日基流出流图 (rows, cols)
        par: 参数向量 (至少16个元素)，torch.Tensor
        sink_classified: 分类土地利用栅格 (rows, cols)，值为1(耕地)、2(自然/草地)、3(不透水)

    返回:
        N_surf_new: 新地表氮库存 (rows, cols)
        N_soil_new: 新土壤氮库存 (rows, cols)
        N_gw_new: 新地下氮库存 (rows, cols)
        L_surf: 地表负荷输出图 (rows, cols)
        L_inter: 壤中流负荷输出图 (rows, cols)
        L_base: 基流负荷输出图 (rows, cols)
        L_out: 总负荷输出图 (rows, cols)
    """
    # 切片氮库参数
    c_surf_release, c_soil_release, c_gw_release, beta_surf_to_soil, beta_soil_to_gw = \
        slice_three_store_nitrogen_params_torch(par, sink_classified)

    # 1. 地表氮库更新
    # 氮输入加入地表库存
    N_surf = N_surf_prev + source_day
    N_surf = torch.clamp_min(N_surf, 0.0)

    # 地表负荷释放
    # 使用稳定指数计算：1 - exp(-c*Q)，当c*Q很小时近似线性
    cQ_surf = c_surf_release * Q_surf
    # 数值稳定：避免exp(-cQ)下溢
    release_factor_surf = 1.0 - torch.exp(-cQ_surf)
    L_surf_raw = N_surf * release_factor_surf
    L_surf = torch.min(L_surf_raw, N_surf)  # 不超过可用库存
    L_surf = torch.clamp_min(L_surf, 0.0)

    # 释放后剩余库存
    N_surf_after_release = N_surf - L_surf
    N_surf_after_release = torch.clamp_min(N_surf_after_release, 0.0)

    # 地表到土壤传递
    to_soil_raw = beta_surf_to_soil * N_surf_after_release
    to_soil = torch.min(to_soil_raw, N_surf_after_release)  # 不超过剩余库存
    to_soil = torch.clamp_min(to_soil, 0.0)

    # 最终地表库存
    N_surf_new = N_surf_after_release - to_soil
    N_surf_new = torch.clamp_min(N_surf_new, 0.0)

    # 2. 土壤氮库更新
    # 来自地表的传递加入土壤库存
    N_soil = N_soil_prev + to_soil
    N_soil = torch.clamp_min(N_soil, 0.0)

    # 壤中流负荷释放
    cQ_soil = c_soil_release * Q_inter
    release_factor_soil = 1.0 - torch.exp(-cQ_soil)
    L_inter_raw = N_soil * release_factor_soil
    L_inter = torch.min(L_inter_raw, N_soil)
    L_inter = torch.clamp_min(L_inter, 0.0)

    # 释放后剩余库存
    N_soil_after_release = N_soil - L_inter
    N_soil_after_release = torch.clamp_min(N_soil_after_release, 0.0)

    # 土壤到地下水传递
    to_gw_raw = beta_soil_to_gw * N_soil_after_release
    to_gw = torch.min(to_gw_raw, N_soil_after_release)
    to_gw = torch.clamp_min(to_gw, 0.0)

    # 最终土壤库存
    N_soil_new = N_soil_after_release - to_gw
    N_soil_new = torch.clamp_min(N_soil_new, 0.0)

    # 3. 地下氮库更新
    # 来自土壤的传递加入地下库存
    N_gw = N_gw_prev + to_gw
    N_gw = torch.clamp_min(N_gw, 0.0)

    # 基流负荷释放
    cQ_gw = c_gw_release * Q_base
    release_factor_gw = 1.0 - torch.exp(-cQ_gw)
    L_base_raw = N_gw * release_factor_gw
    L_base = torch.min(L_base_raw, N_gw)
    L_base = torch.clamp_min(L_base, 0.0)

    # 最终地下库存
    N_gw_new = N_gw - L_base
    N_gw_new = torch.clamp_min(N_gw_new, 0.0)

    # 4. 总负荷输出
    L_out = L_surf + L_inter + L_base
    L_out = torch.clamp_min(L_out, 0.0)

    return N_surf_new, N_soil_new, N_gw_new, L_surf, L_inter, L_base, L_out


def simulate_three_store_nitrogen_torch(source_seq, par, sink_classified,
                                        Q_surf_seq, Q_inter_seq, Q_base_seq,
                                        init_states=None):
    """
    模拟多日三水库氮库过程

    参数:
        source_seq: 氮输入序列 (T, rows, cols)，torch.Tensor
        par: 参数向量 (至少16个元素)，torch.Tensor
        sink_classified: 分类土地利用栅格 (rows, cols)，值为1(耕地)、2(自然/草地)、3(不透水)
        Q_surf_seq: 地表出流序列 (T, rows, cols)
        Q_inter_seq: 壤中流出流序列 (T, rows, cols)
        Q_base_seq: 基流出流序列 (T, rows, cols)
        init_states: 可选初始状态字典，包含键:
            'N_surf': 初始地表氮库存 (rows, cols)
            'N_soil': 初始土壤氮库存 (rows, cols)
            'N_gw': 初始地下氮库存 (rows, cols)
            如果为None，则初始化为零

    返回:
        N_surf_seq: 地表氮库存序列 (T, rows, cols)
        N_soil_seq: 土壤氮库存序列 (T, rows, cols)
        N_gw_seq: 地下氮库存序列 (T, rows, cols)
        L_surf_seq: 地表负荷输出序列 (T, rows, cols)
        L_inter_seq: 壤中流负荷输出序列 (T, rows, cols)
        L_base_seq: 基流负荷输出序列 (T, rows, cols)
        L_out_seq: 总负荷输出序列 (T, rows, cols)
    """
    # 确保输入为torch张量
    if not isinstance(source_seq, torch.Tensor):
        source_seq = torch.as_tensor(source_seq, dtype=torch.float32)
    if not isinstance(par, torch.Tensor):
        par = torch.as_tensor(par, dtype=torch.float32)
    if not isinstance(sink_classified, torch.Tensor):
        sink_classified = torch.as_tensor(sink_classified, dtype=torch.float32)
    if not isinstance(Q_surf_seq, torch.Tensor):
        Q_surf_seq = torch.as_tensor(Q_surf_seq, dtype=torch.float32)
    if not isinstance(Q_inter_seq, torch.Tensor):
        Q_inter_seq = torch.as_tensor(Q_inter_seq, dtype=torch.float32)
    if not isinstance(Q_base_seq, torch.Tensor):
        Q_base_seq = torch.as_tensor(Q_base_seq, dtype=torch.float32)

    device = source_seq.device
    par = par.to(device)
    sink_classified = sink_classified.to(device)
    Q_surf_seq = Q_surf_seq.to(device)
    Q_inter_seq = Q_inter_seq.to(device)
    Q_base_seq = Q_base_seq.to(device)

    # 获取栅格形状
    T, rows, cols = source_seq.shape

    # 初始化状态
    if init_states is None:
        N_surf = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        N_soil = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        N_gw = torch.zeros((rows, cols), dtype=torch.float32, device=device)
    else:
        N_surf = init_states['N_surf'].to(device)
        N_soil = init_states['N_soil'].to(device)
        N_gw = init_states['N_gw'].to(device)

    # 准备输出序列
    N_surf_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    N_soil_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    N_gw_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    L_surf_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    L_inter_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    L_base_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)
    L_out_seq = torch.zeros((T, rows, cols), dtype=torch.float32, device=device)

    # 逐日模拟
    for t in range(T):
        # 获取单日输入
        source_day = source_seq[t]
        Q_surf = Q_surf_seq[t]
        Q_inter = Q_inter_seq[t]
        Q_base = Q_base_seq[t]

        # 调用单步更新
        N_surf_new, N_soil_new, N_gw_new, L_surf, L_inter, L_base, L_out = \
            three_store_nitrogen_step_torch(
                source_day, N_surf, N_soil, N_gw,
                Q_surf, Q_inter, Q_base, par, sink_classified
            )

        # 存储结果
        N_surf_seq[t] = N_surf_new
        N_soil_seq[t] = N_soil_new
        N_gw_seq[t] = N_gw_new
        L_surf_seq[t] = L_surf
        L_inter_seq[t] = L_inter
        L_base_seq[t] = L_base
        L_out_seq[t] = L_out

        # 更新状态
        N_surf = N_surf_new
        N_soil = N_soil_new
        N_gw = N_gw_new

    return N_surf_seq, N_soil_seq, N_gw_seq, L_surf_seq, L_inter_seq, L_base_seq, L_out_seq


def validate_three_store_nitrogen_results(source_seq, N_surf_seq, N_soil_seq, N_gw_seq,
                                          L_surf_seq, L_inter_seq, L_base_seq, L_out_seq,
                                          init_states=None, tolerance=1e-6):
    """
    验证三水库氮库结果

    检查：
    1. 状态非负（氮库存）
    2. 输出非负（负荷输出）
    3. 无NaN/Inf
    4. 质量守恒（氮输入 ≈ 负荷输出总和 + 库存变化）

    参数:
        source_seq: 氮输入序列 (T, rows, cols)，torch.Tensor
        N_surf_seq: 地表氮库存序列 (T, rows, cols)
        N_soil_seq: 土壤氮库存序列 (T, rows, cols)
        N_gw_seq: 地下氮库存序列 (T, rows, cols)
        L_surf_seq: 地表负荷输出序列 (T, rows, cols)
        L_inter_seq: 壤中流负荷输出序列 (T, rows, cols)
        L_base_seq: 基流负荷输出序列 (T, rows, cols)
        L_out_seq: 总负荷输出序列 (T, rows, cols)
        init_states: 初始状态字典，可选，包含键 'N_surf', 'N_soil', 'N_gw'，每个为 (rows, cols) 张量。
                    如果为 None，则假设初始状态为零。
        tolerance: 质量守恒允许的误差容限

    返回:
        validation_dict: 验证结果字典，包含各项检查结果
    """
    # 1. 非负检查
    nonneg_N_surf = torch.all(N_surf_seq >= 0)
    nonneg_N_soil = torch.all(N_soil_seq >= 0)
    nonneg_N_gw = torch.all(N_gw_seq >= 0)
    nonneg_L_surf = torch.all(L_surf_seq >= 0)
    nonneg_L_inter = torch.all(L_inter_seq >= 0)
    nonneg_L_base = torch.all(L_base_seq >= 0)
    nonneg_L_out = torch.all(L_out_seq >= 0)

    # 2. NaN/Inf检查
    def check_finite(tensor):
        return torch.all(torch.isfinite(tensor))

    finite_N_surf = check_finite(N_surf_seq)
    finite_N_soil = check_finite(N_soil_seq)
    finite_N_gw = check_finite(N_gw_seq)
    finite_L_surf = check_finite(L_surf_seq)
    finite_L_inter = check_finite(L_inter_seq)
    finite_L_base = check_finite(L_base_seq)
    finite_L_out = check_finite(L_out_seq)

    # 3. 质量守恒检查
    total_source_input = source_seq.sum()
    total_load_output = L_out_seq.sum()

    if init_states is not None:
        N_total_start = (init_states['N_surf'].sum() +
                         init_states['N_soil'].sum() +
                         init_states['N_gw'].sum())
    else:
        N_total_start = torch.tensor(0.0, device=source_seq.device)

    N_total_end = N_surf_seq[-1].sum() + N_soil_seq[-1].sum() + N_gw_seq[-1].sum()
    delta_N = N_total_end - N_total_start

    # 质量平衡误差
    mass_balance_error = total_source_input - (total_load_output + delta_N)
    mass_balance_abs_error = torch.abs(mass_balance_error)
    mass_conservation = mass_balance_abs_error <= tolerance

    validation_dict = {
        'nonnegative': {
            'N_surf': nonneg_N_surf.item() if nonneg_N_surf.numel() == 1 else nonneg_N_surf,
            'N_soil': nonneg_N_soil.item() if nonneg_N_soil.numel() == 1 else nonneg_N_soil,
            'N_gw': nonneg_N_gw.item() if nonneg_N_gw.numel() == 1 else nonneg_N_gw,
            'L_surf': nonneg_L_surf.item() if nonneg_L_surf.numel() == 1 else nonneg_L_surf,
            'L_inter': nonneg_L_inter.item() if nonneg_L_inter.numel() == 1 else nonneg_L_inter,
            'L_base': nonneg_L_base.item() if nonneg_L_base.numel() == 1 else nonneg_L_base,
            'L_out': nonneg_L_out.item() if nonneg_L_out.numel() == 1 else nonneg_L_out,
        },
        'finite': {
            'N_surf': finite_N_surf.item() if finite_N_surf.numel() == 1 else finite_N_surf,
            'N_soil': finite_N_soil.item() if finite_N_soil.numel() == 1 else finite_N_soil,
            'N_gw': finite_N_gw.item() if finite_N_gw.numel() == 1 else finite_N_gw,
            'L_surf': finite_L_surf.item() if finite_L_surf.numel() == 1 else finite_L_surf,
            'L_inter': finite_L_inter.item() if finite_L_inter.numel() == 1 else finite_L_inter,
            'L_base': finite_L_base.item() if finite_L_base.numel() == 1 else finite_L_base,
            'L_out': finite_L_out.item() if finite_L_out.numel() == 1 else finite_L_out,
        },
        'mass_conservation': mass_conservation.item() if isinstance(mass_conservation, torch.Tensor) else mass_conservation,
        'mass_balance_details': {
            'total_source_input': total_source_input.item(),
            'total_load_output': total_load_output.item(),
            'storage_change': delta_N.item(),
            'mass_balance_error': mass_balance_error.item(),
            'mass_balance_abs_error': mass_balance_abs_error.item(),
        }
    }

    return validation_dict


# =============================================================================
# 7) torch版本三水库三通道氮库核（三路输入）
# =============================================================================

def three_store_nitrogen_step_partitioned_inputs_torch(
    source_day_surf, source_day_soil, source_day_gw,
    N_surf_prev, N_soil_prev, N_gw_prev,
    Q_surf, Q_inter, Q_base, par, sink_classified,
    return_transfers=False
):
    """
    三水库三通道氮库核单步更新（三路输入版本）

    专门处理三路氮输入，分别进入不同氮库：
    - source_day_surf -> N_surf (地表氮库)
    - source_day_soil -> N_soil (土壤氮库)
    - source_day_gw   -> N_gw   (地下氮库)

    保持与第二阶段相同的释放公式：L = N_available * (1 - exp(-c * Q))
    确保非负、库存约束、可微。

    参数:
        source_day_surf: 地表过程氮输入图 (rows, cols) → N_surf
        source_day_soil: BGC过程氮输入图 (rows, cols) → N_soil
        source_day_gw:   HYD过程氮输入图 (rows, cols) → N_gw
        N_surf_prev: 上一时刻地表氮库存 (rows, cols)
        N_soil_prev: 上一时刻土壤氮库存 (rows, cols)
        N_gw_prev: 上一时刻地下氮库存 (rows, cols)
        Q_surf: 地表出流图 (rows, cols)
        Q_inter: 壤中流出流图 (rows, cols)
        Q_base: 基流出流图 (rows, cols)
        par: 物理参数向量 (至少16个元素)，torch.Tensor
        sink_classified: 分类土地利用栅格 (rows, cols)
        return_transfers: 是否返回库间传递量 to_soil, to_gw (默认False)

    返回:
        N_surf_new: 新地表氮库存 (rows, cols)
        N_soil_new: 新土壤氮库存 (rows, cols)
        N_gw_new: 新地下氮库存 (rows, cols)
        L_surf: 地表负荷输出图 (rows, cols)
        L_inter: 壤中流负荷输出图 (rows, cols)
        L_base: 基流负荷输出图 (rows, cols)
        L_out: 总负荷输出图 (rows, cols)
        to_soil: 地表到土壤氮传递量 (rows, cols)，仅当 return_transfers=True 时返回
        to_gw: 土壤到地下水氮传递量 (rows, cols)，仅当 return_transfers=True 时返回
    """
    # 切片氮库参数
    c_surf_release, c_soil_release, c_gw_release, beta_surf_to_soil, beta_soil_to_gw = \
        slice_three_store_nitrogen_params_torch(par, sink_classified)

    # 数值稳定常数
    eps = 1e-8

    # 1. 地表氮库更新
    # 氮输入加入地表库存
    N_surf = N_surf_prev + source_day_surf
    N_surf = torch.clamp_min(N_surf, 0.0)

    # 地表负荷释放
    cQ_surf = c_surf_release * Q_surf
    # 数值稳定：避免exp(-cQ)下溢
    release_factor_surf = 1.0 - torch.exp(-cQ_surf)
    L_surf_raw = N_surf * release_factor_surf
    L_surf = torch.min(L_surf_raw, N_surf)  # 不超过可用库存
    L_surf = torch.clamp_min(L_surf, 0.0)

    # 释放后剩余库存
    N_surf_after_release = N_surf - L_surf
    N_surf_after_release = torch.clamp_min(N_surf_after_release, 0.0)

    # 地表到土壤传递
    to_soil_raw = beta_surf_to_soil * N_surf_after_release
    to_soil = torch.min(to_soil_raw, N_surf_after_release)  # 不超过剩余库存
    to_soil = torch.clamp_min(to_soil, 0.0)

    # 最终地表库存
    N_surf_new = N_surf_after_release - to_soil
    N_surf_new = torch.clamp_min(N_surf_new, 0.0)

    # 2. 土壤氮库更新
    # 来自地表的传递 + BGC过程氮输入
    N_soil = N_soil_prev + source_day_soil + to_soil
    N_soil = torch.clamp_min(N_soil, 0.0)

    # 壤中流负荷释放
    cQ_soil = c_soil_release * Q_inter
    release_factor_soil = 1.0 - torch.exp(-cQ_soil)
    L_inter_raw = N_soil * release_factor_soil
    L_inter = torch.min(L_inter_raw, N_soil)
    L_inter = torch.clamp_min(L_inter, 0.0)

    # 释放后剩余库存
    N_soil_after_release = N_soil - L_inter
    N_soil_after_release = torch.clamp_min(N_soil_after_release, 0.0)

    # 土壤到地下水传递
    to_gw_raw = beta_soil_to_gw * N_soil_after_release
    to_gw = torch.min(to_gw_raw, N_soil_after_release)
    to_gw = torch.clamp_min(to_gw, 0.0)

    # 最终土壤库存
    N_soil_new = N_soil_after_release - to_gw
    N_soil_new = torch.clamp_min(N_soil_new, 0.0)

    # 3. 地下氮库更新
    # 来自土壤的传递 + HYD过程氮输入
    N_gw = N_gw_prev + source_day_gw + to_gw
    N_gw = torch.clamp_min(N_gw, 0.0)

    # 基流负荷释放
    cQ_gw = c_gw_release * Q_base
    release_factor_gw = 1.0 - torch.exp(-cQ_gw)
    L_base_raw = N_gw * release_factor_gw
    L_base = torch.min(L_base_raw, N_gw)
    L_base = torch.clamp_min(L_base, 0.0)

    # 最终地下库存
    N_gw_new = N_gw - L_base
    N_gw_new = torch.clamp_min(N_gw_new, 0.0)

    # 4. 总负荷输出
    L_out = L_surf + L_inter + L_base
    L_out = torch.clamp_min(L_out, 0.0)

    return N_surf_new, N_soil_new, N_gw_new, L_surf, L_inter, L_base, L_out


# =============================================================================
# 8) torch版本局地物理核 + 三水库水文 + 三层氮库 单日整合
# =============================================================================

def run_local_physics_step_with_three_store_nitrogen_torch(
    source, par, f_surface, sink_classified,
    bgc_dis_pool, bgc_ads_pool, hyd_dis_pool, hyd_ads_pool,
    pcp, surface_flow,
    S_surf_prev, S_soil_prev, S_gw_prev,
    N_surf_prev, N_soil_prev, N_gw_prev,
    return_legacy_outputs=False,
    return_hydrology_internals=False
):
    """
    局地物理核 + 三水库水文 + 三层氮库 单日整合函数

    执行顺序：
    1. 调用现有 run_local_physics_step_torch 获取6通道输出
    2. 调用三水库水文步进生成水文出流
    3. 构造三路氮输入（按物理过程映射）
    4. 调用三路输入氮库函数更新氮库并计算负荷输出

    参数:
        source: 源项空间图 (rows, cols)
        par: 物理参数向量 (至少73个元素：现有N + 水文11 + 氮库5)
        f_surface: 地表比例标量 (0~1)
        sink_classified: 分类土地利用栅格 (rows, cols)
        bgc_dis_pool: 初始溶解态BGC池 (rows, cols)
        bgc_ads_pool: 初始吸附态BGC池 (rows, cols)
        hyd_dis_pool: 初始溶解态HYD池 (rows, cols)
        hyd_ads_pool: 初始吸附态HYD池 (rows, cols)
        pcp: 降雨标量 (mm/day)
        surface_flow: 地表流标量 (mm/day)
        S_surf_prev: 上一时刻地表水库库存 (rows, cols)
        S_soil_prev: 上一时刻土壤水库库存 (rows, cols)
        S_gw_prev: 上一时刻地下水水库库存 (rows, cols)
        N_surf_prev: 上一时刻地表氮库存 (rows, cols)
        N_soil_prev: 上一时刻土壤氮库存 (rows, cols)
        N_gw_prev: 上一时刻地下氮库存 (rows, cols)
        return_legacy_outputs: 是否返回旧的6通道输出（用于调试）
        return_hydrology_internals: 是否返回水文层内部诊断量（Q_surf, Q_inter, Q_base, to_soil, to_gw）

    返回:
        L_surf: 地表负荷输出图 (rows, cols)
        L_inter: 壤中流负荷输出图 (rows, cols)
        L_base: 基流负荷输出图 (rows, cols)
        L_out: 总负荷输出图 (rows, cols)
        S_surf_new: 新地表水库库存 (rows, cols)
        S_soil_new: 新土壤水库库存 (rows, cols)
        S_gw_new: 新地下水水库库存 (rows, cols)
        N_surf_new: 新地表氮库存 (rows, cols)
        N_soil_new: 新土壤氮库存 (rows, cols)
        N_gw_new: 新地下氮库存 (rows, cols)
        bgc_dis_pool_new: 更新后溶解态BGC池 (rows, cols)
        bgc_ads_pool_new: 更新后吸附态BGC池 (rows, cols)
        hyd_dis_pool_new: 更新后溶解态HYD池 (rows, cols)
        hyd_ads_pool_new: 更新后吸附态HYD池 (rows, cols)

        如果 return_hydrology_internals=True，额外返回水文层内部诊断量：
        Q_surf, Q_inter, Q_base, to_soil, to_gw

        如果 return_legacy_outputs=True，额外返回旧局地物理核输出：
        surf_dis, surf_ads, bgc_dis_total, bgc_ads_total,
        hyd_dis_total, hyd_ads_total
    """
    # 1. 调用现有局地物理核
    (local_output,
     surf_dis, surf_ads,
     bgc_dis_total, bgc_ads_total,
     hyd_dis_total, hyd_ads_total,
     bgc_dis_pool_new, bgc_ads_pool_new,
     hyd_dis_pool_new, hyd_ads_pool_new) = run_local_physics_step_torch(
        source, par, f_surface, sink_classified,
        bgc_dis_pool, bgc_ads_pool, hyd_dis_pool, hyd_ads_pool,
        pcp, surface_flow
    )

    # 2. 调用三水库水文步进生成水文出流
    # 切片水文参数
    k_surf_grid, k_soil_grid, k_gw_grid, alpha_surf_to_soil, alpha_soil_to_gw = \
        slice_three_store_hydrology_params_torch(par, sink_classified)

    # 计算单日水文出流（同时获取库间传递量）
    S_surf_new, S_soil_new, S_gw_new, Q_surf, Q_inter, Q_base, to_soil, to_gw = \
        three_store_hydrology_step_torch(
            pcp, S_surf_prev, S_soil_prev, S_gw_prev,
            k_surf_grid, k_soil_grid, k_gw_grid,
            alpha_surf_to_soil, alpha_soil_to_gw,
            return_transfers=True
        )

    # 3. 构造三路氮输入（按物理过程映射）
    source_day_surf = surf_dis + surf_ads           # 地表过程 → N_surf
    source_day_soil = bgc_dis_total + bgc_ads_total  # BGC过程 → N_soil
    source_day_gw = hyd_dis_total + hyd_ads_total   # HYD过程 → N_gw

    # 4. 调用三路输入氮库函数
    N_surf_new, N_soil_new, N_gw_new, L_surf, L_inter, L_base, L_out = \
        three_store_nitrogen_step_partitioned_inputs_torch(
            source_day_surf, source_day_soil, source_day_gw,
            N_surf_prev, N_soil_prev, N_gw_prev,
            Q_surf, Q_inter, Q_base, par, sink_classified
        )

    # 5. 返回结果
    if return_legacy_outputs:
        base_tuple = (L_surf, L_inter, L_base, L_out,
                      S_surf_new, S_soil_new, S_gw_new,
                      N_surf_new, N_soil_new, N_gw_new,
                      surf_dis, surf_ads,
                      bgc_dis_total, bgc_ads_total,
                      hyd_dis_total, hyd_ads_total,
                      bgc_dis_pool_new, bgc_ads_pool_new,
                      hyd_dis_pool_new, hyd_ads_pool_new)
    else:
        base_tuple = (L_surf, L_inter, L_base, L_out,
                      S_surf_new, S_soil_new, S_gw_new,
                      N_surf_new, N_soil_new, N_gw_new,
                      bgc_dis_pool_new, bgc_ads_pool_new,
                      hyd_dis_pool_new, hyd_ads_pool_new)

    # 追加水文内部诊断量（如果需要）
    if return_hydrology_internals:
        return base_tuple + (Q_surf, Q_inter, Q_base, to_soil, to_gw)
    else:
        return base_tuple


# =============================================================================
# 9) torch版本单日含拓扑传播训练链路（三水库氮库版）
# =============================================================================

def run_single_day_with_trace_three_store_nitrogen_torch(
    source, par, f_surface, sink_classified,
    sink, slope, dir_up, postorder_nodes, xy,
    bgc_dis_pool, bgc_ads_pool, hyd_dis_pool, hyd_ads_pool,
    S_surf_prev, S_soil_prev, S_gw_prev,
    N_surf_prev, N_soil_prev, N_gw_prev,
    pcp, surface_flow, outlet_code,
    return_legacy_outputs=False, stats=None,
    return_internal_diagnostics=False
):
    """
    torch版本单日含拓扑传播训练链路（三水库氮库版）

    串联新局地物理核（三水库水文+氮库）和空间传播：
    1. 调用 run_local_physics_step_with_three_store_nitrogen_torch 获取负荷输出和更新状态
    2. 将总负荷输出 L_out 送入 trace_back_iterative_torch（统一路由，不区分形态）
    3. 用 outlet_code 取出口值 out_nitrogen
    4. 定义 day_output = out_nitrogen（仅含路由后总负荷）

    参数:
        source: 单日源项空间图 (rows, cols)，torch.Tensor
        par: 物理参数向量 (至少73个元素)，torch.Tensor，requires_grad=True
        f_surface: 单个标量，torch.Tensor
        sink_classified: 分类后的土地利用图 (rows, cols)，torch.Tensor
        sink: 原始 landuse 图 (rows, cols)，numpy 或 torch
        slope: 坡度图 (rows, cols)，numpy 或 torch
        dir_up: 流向拓扑字典
        postorder_nodes: 后序遍历节点列表
        xy: 节点坐标字典
        bgc_dis_pool: 初始溶解态 bgc 池 (rows, cols)，torch.Tensor
        bgc_ads_pool: 初始吸附态 bgc 池 (rows, cols)，torch.Tensor
        hyd_dis_pool: 初始溶解态 hyd 池 (rows, cols)，torch.Tensor
        hyd_ads_pool: 初始吸附态 hyd 池 (rows, cols)，torch.Tensor
        S_surf_prev: 上一时刻地表水库库存 (rows, cols)，torch.Tensor
        S_soil_prev: 上一时刻土壤水库库存 (rows, cols)，torch.Tensor
        S_gw_prev: 上一时刻地下水水库库存 (rows, cols)，torch.Tensor
        N_surf_prev: 上一时刻地表氮库存 (rows, cols)，torch.Tensor
        N_soil_prev: 上一时刻土壤氮库存 (rows, cols)，torch.Tensor
        N_gw_prev: 上一时刻地下氮库存 (rows, cols)，torch.Tensor
        pcp: 单日降雨标量，torch.Tensor
        surface_flow: 单日地表流标量，torch.Tensor
        outlet_code: 出口编码字符串，例如 cfg["OUTLET_CODE"]
        stats: 可选字典，用于收集衰减统计信息，传递给 trace_back_iterative_torch
        return_legacy_outputs: 是否返回旧局地物理核输出（用于调试）
        return_internal_diagnostics: 是否返回水文层内部诊断量（Q_surf, Q_inter, Q_base, to_soil, to_gw）

    返回:
        day_output: 单日总输出（标量，即路由后总负荷），torch.Tensor
        out_nitrogen: 出口总负荷值（标量），torch.Tensor
        L_out: 总负荷输出图 (rows, cols)，路由前，torch.Tensor
        L_out_map: 路由后总负荷图 (rows, cols)，torch.Tensor
        S_surf_new: 新地表水库库存 (rows, cols)，torch.Tensor
        S_soil_new: 新土壤水库库存 (rows, cols)，torch.Tensor
        S_gw_new: 新地下水水库库存 (rows, cols)，torch.Tensor
        N_surf_new: 新地表氮库存 (rows, cols)，torch.Tensor
        N_soil_new: 新土壤氮库存 (rows, cols)，torch.Tensor
        N_gw_new: 新地下氮库存 (rows, cols)，torch.Tensor

        如果 return_internal_diagnostics=True，额外返回水文层内部诊断量：
        Q_surf, Q_inter, Q_base, to_soil, to_gw

        如果 return_legacy_outputs=True，额外返回旧局地物理核输出：
        surf_dis, surf_ads, bgc_dis_total, bgc_ads_total,
        hyd_dis_total, hyd_ads_total,
        bgc_dis_pool_new, bgc_ads_pool_new,
        hyd_dis_pool_new, hyd_ads_pool_new
    """
    # 1. 调用新局地物理核（三水库水文+氮库）
    if return_legacy_outputs:
        # 调用下层函数，可能返回水文内部诊断量
        result = run_local_physics_step_with_three_store_nitrogen_torch(
            source, par, f_surface, sink_classified,
            bgc_dis_pool, bgc_ads_pool, hyd_dis_pool, hyd_ads_pool,
            pcp, surface_flow,
            S_surf_prev, S_soil_prev, S_gw_prev,
            N_surf_prev, N_soil_prev, N_gw_prev,
            return_legacy_outputs=True,
            return_hydrology_internals=return_internal_diagnostics
        )
        # 解包结果：基础部分 + 可能的水文内部量
        if return_internal_diagnostics:
            (_, _, _, L_out,
             S_surf_new, S_soil_new, S_gw_new,
             N_surf_new, N_soil_new, N_gw_new,
             surf_dis, surf_ads,
             bgc_dis_total, bgc_ads_total,
             hyd_dis_total, hyd_ads_total,
             bgc_dis_pool_new, bgc_ads_pool_new,
             hyd_dis_pool_new, hyd_ads_pool_new,
             Q_surf, Q_inter, Q_base, to_soil, to_gw) = result
        else:
            (_, _, _, L_out,
             S_surf_new, S_soil_new, S_gw_new,
             N_surf_new, N_soil_new, N_gw_new,
             surf_dis, surf_ads,
             bgc_dis_total, bgc_ads_total,
             hyd_dis_total, hyd_ads_total,
             bgc_dis_pool_new, bgc_ads_pool_new,
             hyd_dis_pool_new, hyd_ads_pool_new) = result
    else:
        # 调用下层函数，可能返回水文内部诊断量
        result = run_local_physics_step_with_three_store_nitrogen_torch(
            source, par, f_surface, sink_classified,
            bgc_dis_pool, bgc_ads_pool, hyd_dis_pool, hyd_ads_pool,
            pcp, surface_flow,
            S_surf_prev, S_soil_prev, S_gw_prev,
            N_surf_prev, N_soil_prev, N_gw_prev,
            return_legacy_outputs=False,
            return_hydrology_internals=return_internal_diagnostics
        )
        # 解包结果：基础部分 + 可能的水文内部量
        if return_internal_diagnostics:
            (_, _, _, L_out,
             S_surf_new, S_soil_new, S_gw_new,
             N_surf_new, N_soil_new, N_gw_new,
             bgc_dis_pool_new, bgc_ads_pool_new,
             hyd_dis_pool_new, hyd_ads_pool_new,
             Q_surf, Q_inter, Q_base, to_soil, to_gw) = result
        else:
            (_, _, _, L_out,
             S_surf_new, S_soil_new, S_gw_new,
             N_surf_new, N_soil_new, N_gw_new,
             bgc_dis_pool_new, bgc_ads_pool_new,
             hyd_dis_pool_new, hyd_ads_pool_new) = result

    # 2. 将总负荷输出 L_out 送入 trace_back_iterative_torch
    # 注意：trace_back_iterative_torch 需要 mode 参数，但 L_out 是总负荷，不区分溶解/吸附态。
    # 此处暂时使用 mode="dis"（任意选择，因为衰减公式可能依赖 mode）。
    # 若需要更精细处理，后续可调整衰减函数使其不依赖 mode，或传入一个默认 mode。
    L_out_map = trace_back_iterative_torch(
        L_out, dir_up, postorder_nodes, xy,
        sink, slope, par, mode="dis", stats=stats
    )

    # 3. 用 outlet_code 取出口值
    oi, oj = int(outlet_code[:4]), int(outlet_code[4:])
    out_nitrogen = L_out_map[oi, oj]

    # 4. 构造单日总输出（仅含路由后总负荷）
    day_output = out_nitrogen

    # 5. 返回结果
    if return_legacy_outputs:
        base_tuple = (day_output, out_nitrogen, L_out, L_out_map,
                      S_surf_new, S_soil_new, S_gw_new,
                      N_surf_new, N_soil_new, N_gw_new,
                      surf_dis, surf_ads,
                      bgc_dis_total, bgc_ads_total,
                      hyd_dis_total, hyd_ads_total,
                      bgc_dis_pool_new, bgc_ads_pool_new,
                      hyd_dis_pool_new, hyd_ads_pool_new)
    else:
        base_tuple = (day_output, out_nitrogen, L_out, L_out_map,
                      S_surf_new, S_soil_new, S_gw_new,
                      N_surf_new, N_soil_new, N_gw_new,
                      bgc_dis_pool_new, bgc_ads_pool_new,
                      hyd_dis_pool_new, hyd_ads_pool_new)

    # 追加水文内部诊断量（如果需要）
    if return_internal_diagnostics:
        return base_tuple + (Q_surf, Q_inter, Q_base, to_soil, to_gw)
    else:
        return base_tuple


# =============================================================================
# 模块导出
# =============================================================================
def build_source_maps_with_ml_torch(X, groups, idx_g, pix_r, pix_c, group_ratio,
                                     landuse_shape, L_daily, ml_w, ml_b, device='cpu'):
    if not isinstance(X, torch.Tensor):
        X = torch.as_tensor(X, dtype=torch.float32, device=device)
    if not isinstance(groups, torch.Tensor):
        groups = torch.as_tensor(groups, dtype=torch.long, device=device)
    if not isinstance(pix_r, torch.Tensor):
        pix_r = torch.as_tensor(pix_r, dtype=torch.long, device=device)
    if not isinstance(pix_c, torch.Tensor):
        pix_c = torch.as_tensor(pix_c, dtype=torch.long, device=device)
    if not isinstance(L_daily, torch.Tensor):
        L_daily = torch.as_tensor(L_daily, dtype=torch.float32, device=device)

    rows, cols = landuse_shape
    logits = X @ ml_w + ml_b
    rc_by_g = {}
    weights_by_g = {}
    ratio_by_g = {}
    for g, inds in idx_g.items():
        if not isinstance(inds, torch.Tensor):
            inds = torch.as_tensor(inds, dtype=torch.long, device=device)
        rc_by_g[g] = (pix_r[inds], pix_c[inds])
        ratio_by_g[g] = torch.tensor(group_ratio.get(int(g), 0.0), dtype=torch.float32, device=device)
        group_logits = logits[inds]
        group_logits = group_logits - torch.max(group_logits)
        weights_by_g[g] = torch.softmax(group_logits, dim=0)

    source_maps = []
    for t in range(len(L_daily)):
        total = L_daily[t]
        src = torch.zeros((rows, cols), dtype=torch.float32, device=device)
        for g, (rr, cc) in rc_by_g.items():
            if group_ratio.get(int(g), 0.0) <= 0.0:
                continue
            src[rr, cc] = total * ratio_by_g[g] * weights_by_g[g]
        source_maps.append(src)
    return source_maps


__all__ = [
    'softplus_torch',
    'sigmoid_torch',
    'group_softmax_torch',
    'LearnableDailyTotalTorch',
    'build_source_maps_with_ml_torch',
    'par_allocation_torch',
    'path_allocation_day_torch',
    'bgc_legacy_contribution_torch',
    'hyd_legacy_step_torch',
    '_attenuate_incremental_torch',
    'trace_back_iterative_torch',
    'run_local_physics_step_torch',
    'run_single_day_with_trace_torch',
    'slice_three_store_hydrology_params_torch',
    'three_store_hydrology_step_torch',
    'simulate_three_store_hydrology_torch',
    'validate_three_store_hydrology_results',
    'slice_three_store_nitrogen_params_torch',
    'three_store_nitrogen_step_torch',
    'simulate_three_store_nitrogen_torch',
    'validate_three_store_nitrogen_results',
    'three_store_nitrogen_step_partitioned_inputs_torch',
    'run_local_physics_step_with_three_store_nitrogen_torch',
    'run_single_day_with_trace_three_store_nitrogen_torch'
]
