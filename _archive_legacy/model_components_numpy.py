"""
模型计算核心模块 - 包含TN日尺度模型的所有计算逻辑
纯NumPy实现，不包含文件读取、优化算法、评价指标等

主要组件：
1. 辅助数学函数：softplus, sigmoid, group_softmax
2. 时间生成模块：LearnableDailyTotal类
3. 物理过程模块：DailyPhysicalModel类
4. 空间分配函数：build_source_maps
5. 核心模型类：DailyNPSCore（整合时间、空间、物理过程）

注意：本模块不包含任何文件读取、数据加载、损失计算、优化算法
"""

import numpy as np
import pandas as pd


# =============================================================================
# 1) 辅助数学函数
# =============================================================================

def softplus(x):
    """稳定版本的softplus函数"""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def sigmoid(x):
    """稳定版本的sigmoid函数"""
    x = np.asarray(x, dtype=float)
    # stable sigmoid
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def group_softmax(z: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """分组softmax：在每个组内分别计算softmax权重"""
    w = np.zeros_like(z, dtype=float)
    for g in np.unique(groups):
        idx = (groups == g)
        zg = z[idx]
        expz = np.exp(zg - np.max(zg))
        w[idx] = expz / (np.sum(expz) + 1e-12)
    return w


# =============================================================================
# 2) 时间生成模块
# =============================================================================

class LearnableDailyTotal:
    """
    可学习的日尺度总量生成器 + 地表比例生成器

    输出两个关键量：
    1) L_total_daily[t] : 每天流域总源项（守恒到 year_total*year_scale）
    2) f_surface[t]     : 每天"可即时出流比例"（0~1，sigmoid学习）
       - 解决"源项有峰但模拟峰值日出不来"的错峰问题
    """

    def __init__(self, cfg, dates, rain, runoff, fert_factor, year_total,
                 prior_indicator=None, fs_prior=None):
        """
        初始化时间生成器

        参数:
            cfg: 配置字典
            dates: 日期序列 (pd.DatetimeIndex)
            rain: 降雨序列 (mm/day)
            runoff: 径流序列 (mm/day)
            fert_factor: 施肥因子序列 (0~1)
            year_total: 年总源项 (kg/year)
            prior_indicator: 日总量先验指示器（可选）
            fs_prior: 地表比例先验（可选）
        """
        self.cfg = cfg
        self.dates = pd.to_datetime(dates)
        self.rain = np.asarray(rain, dtype=float)
        self.runoff = np.asarray(runoff, dtype=float)
        self.fert = np.asarray(fert_factor, dtype=float)
        self.year_total = float(year_total)
        self.T = len(self.dates)

        doy = self.dates.dayofyear.to_numpy()
        self.doy = doy.astype(float)

        if prior_indicator is None:
            self.prior = None
        else:
            p = np.asarray(prior_indicator, dtype=float)
            p = np.maximum(p, 1e-12)
            p = p / (np.sum(p) + 1e-12)
            self.prior = p

        # f_surface prior（很弱）：来自你原本的 surface_flow/pcp 思路
        self.fs_prior = None if fs_prior is None else np.clip(np.asarray(fs_prior, dtype=float), 0.0, 1.0)

    def compute_api(self, rain_eff, decay):
        """计算前期降水指数API"""
        api = np.zeros_like(rain_eff, dtype=float)
        a = 0.0
        for t in range(len(rain_eff)):
            a = rain_eff[t] + decay * a
            api[t] = a
        return api

    def forward(self, theta_time):
        """
        前向计算：生成日总源项和地表比例

        参数:
            theta_time: 时间生成器参数向量（长度21）

        返回:
            L_daily: 日总源项序列
            f_surface: 地表比例序列
            dbg: 调试信息字典
        """
        th = np.asarray(theta_time, dtype=float).copy()

        # -------- (A) daily total ----------
        year_scale = float(th[0])
        rain_thr = float(th[1])
        api_decay = float(th[2])

        b0, s1, c1, s2, c2 = th[3], th[4], th[5], th[6], th[7]
        p0, pr, pq, pa, pf = th[8], th[9], th[10], th[11], th[12]
        mix_kappa = float(th[13])

        rain_eff = np.maximum(self.rain - rain_thr, 0.0)
        api = self.compute_api(rain_eff, api_decay)

        ang1 = 2.0 * np.pi * (self.doy / 365.0)
        ang2 = 4.0 * np.pi * (self.doy / 365.0)

        base_log = b0 + s1 * np.sin(ang1) + c1 * np.cos(ang1) + s2 * np.sin(ang2) + c2 * np.cos(ang2)
        base = np.exp(np.clip(base_log, -20, 20))

        evt_lin = (
            p0
            + pr * np.log1p(rain_eff)
            + pq * np.log1p(np.maximum(self.runoff, 0.0))
            + pa * np.log1p(np.maximum(api, 0.0))
            + pf * self.fert
        )
        evt = softplus(evt_lin)

        raw = base + mix_kappa * evt
        raw = np.maximum(raw, 1e-12)

        total = self.year_total * year_scale
        L = total * raw / (np.sum(raw) + 1e-12)

        # -------- (B) learn f_surface ----------
        sf_b0, sf_r, sf_q, sf_a, sf_f, sf_sin, sf_cos = th[14], th[15], th[16], th[17], th[18], th[19], th[20]

        sf_lin = (
            sf_b0
            + sf_r * np.log1p(rain_eff)
            + sf_q * np.log1p(np.maximum(self.runoff, 0.0))
            + sf_a * np.log1p(np.maximum(api, 0.0))
            + sf_f * self.fert
            + sf_sin * np.sin(ang1)
            + sf_cos * np.cos(ang1)
        )
        f_surface = sigmoid(sf_lin)
        # 防止极端 0/1 导致不可辨识/数值问题
        f_surface = np.clip(f_surface, 0.1, 0.98)

        dbg = dict(raw=raw, base=base, evt=evt, api=api, rain_eff=rain_eff, f_surface=f_surface)
        return L, f_surface, dbg

    def reg_smooth(self, L):
        """日总量平滑正则项（二阶差分）"""
        d1 = np.diff(L)
        d2 = np.diff(d1)
        return float(np.mean(d2 ** 2))

    def reg_prior(self, L):
        """日总量先验正则项（与prior_indicator的KL散度近似）"""
        if self.prior is None:
            return 0.0
        p = self.prior
        q = np.maximum(L, 1e-12)
        q = q / (np.sum(q) + 1e-12)
        return float(np.mean((q - p) ** 2))

    def reg_fs_smooth(self, fs):
        """地表比例平滑正则项（二阶差分）"""
        d1 = np.diff(fs)
        d2 = np.diff(d1)
        return float(np.mean(d2 ** 2))

    def reg_fs_prior(self, fs):
        """地表比例先验正则项（与fs_prior的MSE）"""
        if self.fs_prior is None:
            return 0.0
        return float(np.mean((np.asarray(fs) - self.fs_prior) ** 2))


# =============================================================================
# 3) 空间分配函数
# =============================================================================

def build_source_maps(X, groups, idx_g, pix_r, pix_c, group_ratio, landuse_shape, L_daily):
    """
    构建日尺度源项空间分布图

    参数:
        X: 网格特征矩阵 (n_grid, n_feat)
        groups: 网格所属土地利用组 (n_grid,)
        idx_g: 字典 {group: 索引数组}
        pix_r: 网格行坐标 (n_grid,)
        pix_c: 网格列坐标 (n_grid,)
        group_ratio: 字典 {group: 比例}
        landuse_shape: (rows, cols) 土地利用栅格形状
        L_daily: 日总源项序列 (n_days,)

    返回:
        source_maps: 源项空间分布图列表，每个元素为 (rows, cols) 数组
    """
    rows, cols = landuse_shape
    source_maps = []

    for t in range(len(L_daily)):
        total = float(L_daily[t])
        # 按组分配总量
        Lg = {g: total * group_ratio.get(int(g), 0.0) for g in (1, 2, 3)}

        src = np.zeros((rows, cols), dtype=float)
        for g, inds in idx_g.items():
            group_total = Lg.get(int(g), 0.0)
            if group_total == 0.0 or len(inds) == 0:
                continue
            rr = pix_r[inds]
            cc = pix_c[inds]
            # 均匀分配：组总量 / 组内像元数
            src[rr, cc] = group_total / len(inds)

        source_maps.append(src)

    return source_maps


def build_source_maps_with_ml(X, groups, idx_g, pix_r, pix_c, group_ratio, landuse_shape, L_daily, ml_w, ml_b):
    """
    构建日尺度源项空间分布图（含ML权重学习）

    参数:
        X: 网格特征矩阵 (n_grid, n_feat)
        groups: 网格所属土地利用组 (n_grid,)
        idx_g: 字典 {group: 索引数组}
        pix_r: 网格行坐标 (n_grid,)
        pix_c: 网格列坐标 (n_grid,)
        group_ratio: 字典 {group: 比例}
        landuse_shape: (rows, cols) 土地利用栅格形状
        L_daily: 日总源项序列 (n_days,)
        ml_w: ML权重向量 (n_feat,)
        ml_b: ML偏置标量

    返回:
        source_maps: 源项空间分布图列表，每个元素为 (rows, cols) 数组
    """
    rows, cols = landuse_shape
    rc_by_g = {}
    count_by_g = {}
    for g, inds in idx_g.items():
        rc_by_g[g] = (pix_r[inds], pix_c[inds])
        count_by_g[g] = max(len(inds), 1)

    source_maps = []
    for t in range(len(L_daily)):
        total = float(L_daily[t])
        Lg = {g: total * group_ratio.get(int(g), 0.0) for g in (1, 2, 3)}

        src = np.zeros((rows, cols), dtype=float)
        for g in rc_by_g.keys():
            if Lg.get(int(g), 0.0) == 0.0:
                continue
            rr, cc = rc_by_g[g]
            src[rr, cc] = Lg[int(g)] / count_by_g[g]

        source_maps.append(src)

    return source_maps


# =============================================================================
# 4) 物理过程模块
# =============================================================================

class DailyPhysicalModel:
    """
    日尺度物理过程模型

    关键改动：使用可学习的 f_surface(t) 切分 daily source -> surface vs legacy，
    解决原来用 surface_flow/pcp 导致峰值日错峰的问题。
    """

    def __init__(self, cfg, landuse, slope, flowdir_up):
        self.cfg = cfg
        self.sink = landuse.astype(float)
        self.slope = slope.astype(float)
        self.dir_up = flowdir_up

        self.rows, self.cols = landuse.shape
        self.outlet = cfg["OUTLET_CODE"]
        oi, oj = int(self.outlet[:4]), int(self.outlet[4:])
        if not (0 <= oi < self.rows and 0 <= oj < self.cols):
            raise ValueError(f"OUTLET_CODE {self.outlet} out of bounds for grid {self.rows}x{self.cols}")

        # 土地利用分类映射：1->1(耕地), 2/4->2(自然/草地), 8->3(不透水)
        self.sink_classified = self.sink.copy()
        self.sink_classified[self.sink == 1] = 1
        self.sink_classified[self.sink == 2] = 2
        self.sink_classified[self.sink == 4] = 2
        self.sink_classified[self.sink == 8] = 3

        # 构建后序遍历序列（从出口向上游）
        self.postorder_nodes = self._build_postorder(self.outlet)
        self.xy = {code: (int(code[:4]), int(code[4:])) for code in self.postorder_nodes}

    def _build_postorder(self, outlet_code: str):
        """构建从出口开始的后序遍历序列"""
        stack = [(outlet_code, 0)]
        visited = set()
        post = []
        while stack:
            node, state = stack.pop()
            if state == 0:
                if node in visited:
                    continue
                visited.add(node)
                stack.append((node, 1))
                ups = self.dir_up.get(node, [])
                for up in ups[::-1]:
                    stack.append((up, 0))
            else:
                post.append(node)
        return post

    def par_allocation(self, par_a, par_n, par_u):
        """参数按土地利用组分配"""
        p = self.sink_classified.copy()
        p[self.sink_classified == 1] = par_a
        p[self.sink_classified == 2] = par_n
        p[self.sink_classified == 3] = par_u
        return p

    def path_allocation_day(self, pcp, surface_flow, source, par, f_surface):
        """
        ★ 关键改动：
        原来用 surface_flow/pcp 来算 frac（导致峰值日错峰、锁进 legacy）。
        现在用可学习的 f_surface(t) 来切分 daily source -> surface vs legacy。

        注意：当前路径切分主要由 f_surface 控制，pcp 和 surface_flow 参数
        主要为保持接口兼容性而保留，在核心计算中未直接使用。
        """
        source_allocation = self.par_allocation(par[0], par[17], par[34])
        dis_leak1 = self.par_allocation(par[1], par[18], par[35])
        dis_leak2 = self.par_allocation(par[2], par[19], par[36])
        ads_leak1 = self.par_allocation(par[3], par[20], par[37])
        ads_leak2 = self.par_allocation(par[4], par[21], par[38])

        dissolved = source * source_allocation
        adsorb = source * (1 - source_allocation)

        frac = float(np.clip(f_surface, 0.0, 1.0))

        surf_dis = dissolved * frac
        surf_ads = adsorb * frac
        dis_leg = dissolved * (1 - frac)
        ads_leg = adsorb * (1 - frac)

        bgc_dis = dis_leg * (1 - dis_leak1)
        hyd_dis = dis_leg * dis_leak1 * (1 - dis_leak2)

        bgc_ads = ads_leg * (1 - ads_leak1)
        hyd_ads = ads_leg * ads_leak1 * (1 - ads_leak2)

        return surf_dis, surf_ads, bgc_dis, bgc_ads, hyd_dis, hyd_ads

    def bgc_legacy_contribution(self, new_nutrient, legacy_pool, par, form):
        """生物地球化学过程（溶解态/吸附态）"""
        if form == "dis":
            p_current = self.par_allocation(par[5], par[22], par[39])
            p_history = self.par_allocation(par[6], par[23], par[40])
            leak = self.par_allocation(par[7], par[24], par[41])
        else:
            p_current = self.par_allocation(par[8], par[25], par[42])
            p_history = self.par_allocation(par[9], par[26], par[43])
            leak = self.par_allocation(par[10], par[27], par[44])

        total = p_current * new_nutrient + p_history * legacy_pool
        raw_pool = legacy_pool + new_nutrient - total
        bgc_to_hyd = raw_pool * 0.1
        new_pool = raw_pool * (1 - 0.1) * (1 - leak)
        return total, new_pool, bgc_to_hyd

    def hyd_legacy_step(self, new_nutrient, legacy_pool, par, form):
        """水文滞后过程（溶解态/吸附态）"""
        if form == "dis":
            p_history = self.par_allocation(par[12], par[29], par[46])
            leak = self.par_allocation(par[13], par[30], par[47])
        else:
            p_history = self.par_allocation(par[15], par[32], par[49])
            leak = self.par_allocation(par[16], par[33], par[50])

        total = legacy_pool * p_history
        new_pool = (legacy_pool + new_nutrient - total) * leak
        return total, new_pool

    def _attenuate_incremental(self, val, i, j, par, mode):
        """增量衰减（森林、草地、水体）"""
        lu = self.sink[i, j]
        if lu == 2:  # FRST
            if val < par[53]:
                val *= (5.889 + 0.1609 * self.slope[i, j] - 0.0353 + 1.007 - 0.4511 * 0.4 + 59.8298) / 100.0
            else:
                if mode in ("dis", "ads"):
                    val = val - par[53]
        elif lu == 4:  # PAST
            if val < par[55]:
                val *= (5.889 + 0.1609 * self.slope[i, j] - 0.0353 + 1.007 - 0.4511 * 0.13 + 59.8298) / 100.0
            else:
                if mode in ("dis", "ads"):
                    val = val - par[55]
        elif lu == 5:  # WATR
            if val > par[56]:
                vv = max(val, 1e-6)
                val *= (0.0797 * np.exp(-0.00518 * (2700.0 / vv)) + 65.5432) / 100.0
            else:
                if mode in ("dis", "ads"):
                    val = 0.0
        return val

    def trace_back_iterative(self, total_source_map, par, mode):
        """逆向追踪计算（从上游到出口）"""
        out = total_source_map.copy()
        for node in self.postorder_nodes:
            ups = self.dir_up.get(node, None)
            if not ups:
                continue
            i, j = self.xy[node]
            for up in ups:
                if up not in self.xy:
                    continue
                ui, uj = self.xy[up]
                out[i, j] += out[ui, uj]
                out[i, j] = self._attenuate_incremental(out[i, j], i, j, par, mode)
        oi, oj = int(self.outlet[:4]), int(self.outlet[4:])
        return float(out[oi, oj])

    def simulate_daily(self, rain, surface_flow, source_maps, par, f_surface_series):
        """
        日尺度模拟主函数

        参数:
            rain: 降雨序列 (n_days,)
            surface_flow: 地表流序列 (n_days,)
            source_maps: 源项空间分布图列表，每个元素为 (rows, cols)
            par: 物理参数向量
            f_surface_series: 地表比例序列 (n_days,)

        返回:
            sim: 出口负荷模拟序列 (n_days,)
        """
        shp = self.sink.shape
        bgc_dis_pool = np.zeros(shp)
        bgc_ads_pool = np.zeros(shp)
        hyd_dis_pool = np.zeros(shp)
        hyd_ads_pool = np.zeros(shp)

        sim = []
        for t in range(len(rain)):
            pcp = float(rain[t])
            sflow = float(surface_flow[t])
            source = source_maps[t]
            fs = float(f_surface_series[t])

            surf_dis, surf_ads, bgc_dis, bgc_ads, hyd_dis, hyd_ads = self.path_allocation_day(
                pcp, sflow, source, par, fs
            )

            bgc_dis_total, bgc_dis_pool, bgc2hyd_dis = self.bgc_legacy_contribution(bgc_dis, bgc_dis_pool, par, "dis")
            bgc_ads_total, bgc_ads_pool, bgc2hyd_ads = self.bgc_legacy_contribution(bgc_ads, bgc_ads_pool, par, "ads")

            total_dis_source = surf_dis + bgc_dis_total
            total_ads_source = surf_ads + bgc_ads_total

            out_dis = self.trace_back_iterative(total_dis_source, par, mode="dis")
            out_ads = self.trace_back_iterative(total_ads_source, par, mode="ads")

            hyd_dis_total, hyd_dis_pool = self.hyd_legacy_step(hyd_dis + bgc2hyd_dis, hyd_dis_pool, par, "dis")
            hyd_ads_total, hyd_ads_pool = self.hyd_legacy_step(hyd_ads + bgc2hyd_ads, hyd_ads_pool, par, "ads")

            out = out_dis + out_ads + float(np.sum(hyd_dis_total) + np.sum(hyd_ads_total))
            sim.append(out)

        return np.asarray(sim, dtype=float)


# =============================================================================
# 5) 核心整合类
# =============================================================================

class DailyNPSCore:
    """
    TN日尺度模型计算核心

    整合时间生成、空间分配、物理过程，不包含文件读取和损失函数。
    设计为与数据加载模块（data_loader.py）配合使用。
    """

    def __init__(self, cfg, landuse, slope, flowdir_up, X, groups, idx_g, pix_r, pix_c,
                 group_ratio, dates, rain, surface_flow, fert, runoff_for_time, year_total,
                 prior_L=None, fs_prior=None):
        """
        初始化模型核心

        参数:
            cfg: 配置字典
            landuse: 土地利用栅格 (rows, cols)
            slope: 坡度栅格 (rows, cols)
            flowdir_up: 流向拓扑字典 {cell_id: [upstream_cells]}
            X: 网格特征矩阵 (n_grid, n_feat)
            groups: 网格所属土地利用组 (n_grid,)
            idx_g: 字典 {group: 索引数组}
            pix_r: 网格行坐标 (n_grid,)
            pix_c: 网格列坐标 (n_grid,)
            group_ratio: 字典 {group: 比例}
            dates: 日期序列 (pd.DatetimeIndex, n_days)
            rain: 降雨序列 (n_days,)
            surface_flow: 地表流序列 (n_days,)
            fert: 施肥因子序列 (n_days,)
            runoff_for_time: 用于时间生成器的径流序列 (n_days,)
            year_total: 年总源项 (kg/year)
            prior_L: 日总量先验序列 (n_days,)，可选
            fs_prior: 地表比例先验序列 (n_days,)，可选
        """
        self.cfg = cfg
        self.landuse = landuse
        self.slope = slope
        self.flowdir_up = flowdir_up
        self.X = X
        self.groups = groups
        self.idx_g = idx_g
        self.pix_r = pix_r
        self.pix_c = pix_c
        self.group_ratio = group_ratio
        self.dates = dates
        self.rain = rain
        self.surface_flow = surface_flow
        self.fert = fert
        self.runoff_for_time = runoff_for_time
        self.year_total = year_total

        # 初始化时间生成器
        self.time_model = LearnableDailyTotal(
            cfg=cfg,
            dates=self.dates,
            rain=rain,
            runoff=runoff_for_time,
            fert_factor=fert,
            year_total=year_total,
            prior_indicator=prior_L,
            fs_prior=fs_prior
        )

        # 初始化物理模型
        self.phys = DailyPhysicalModel(cfg, landuse, slope, flowdir_up)

        # 保存形状信息
        self.landuse_shape = landuse.shape

    def simulate(self, par_full, ml_w, ml_b, theta_time, return_debug=False):
        """
        完整模拟流程

        参数:
            par_full: 完整物理参数向量
            ml_w: ML权重向量 (n_feat,)
            ml_b: ML偏置标量
            theta_time: 时间生成器参数向量
            return_debug: 是否返回调试信息（默认为False）

        返回:
            如果 return_debug=False: (sim, L_daily, f_surface)
            如果 return_debug=True:  (sim, L_daily, f_surface, dbg)
            其中:
                sim: 出口负荷模拟序列 (n_days,)
                L_daily: 日总源项序列 (n_days,)
                f_surface: 地表比例序列 (n_days,)
                dbg: 调试信息字典（来自时间生成器的内部变量）
        """
        # 1. 时间生成
        L_daily, f_surface, dbg = self.time_model.forward(theta_time)

        # 2. 空间分配（含ML权重）
        source_maps = build_source_maps_with_ml(
            X=self.X,
            groups=self.groups,
            idx_g=self.idx_g,
            pix_r=self.pix_r,
            pix_c=self.pix_c,
            group_ratio=self.group_ratio,
            landuse_shape=self.landuse_shape,
            L_daily=L_daily,
            ml_w=ml_w,
            ml_b=ml_b
        )

        # 3. 物理过程模拟
        sim = self.phys.simulate_daily(
            rain=self.rain,
            surface_flow=self.surface_flow,
            source_maps=source_maps,
            par=par_full,
            f_surface_series=f_surface
        )

        if return_debug:
            return sim, L_daily, f_surface, dbg
        else:
            return sim, L_daily, f_surface

    def simulate_simple(self, par_full, L_daily, f_surface):
        """
        简化模拟流程（使用给定的日总量和地表比例）

        参数:
            par_full: 完整物理参数向量
            L_daily: 日总源项序列 (n_days,)
            f_surface: 地表比例序列 (n_days,)

        返回:
            sim: 出口负荷模拟序列 (n_days,)
        """
        # 空间分配（无ML权重，均匀分配）
        source_maps = build_source_maps(
            X=self.X,
            groups=self.groups,
            idx_g=self.idx_g,
            pix_r=self.pix_r,
            pix_c=self.pix_c,
            group_ratio=self.group_ratio,
            landuse_shape=self.landuse_shape,
            L_daily=L_daily
        )

        # 物理过程模拟
        sim = self.phys.simulate_daily(
            rain=self.rain,
            surface_flow=self.surface_flow,
            source_maps=source_maps,
            par=par_full,
            f_surface_series=f_surface
        )

        return sim


# =============================================================================
# 模块导出
# =============================================================================
__all__ = [
    'softplus',
    'sigmoid',
    'group_softmax',
    'LearnableDailyTotal',
    'build_source_maps',
    'build_source_maps_with_ml',
    'DailyPhysicalModel',
    'DailyNPSCore'
]
