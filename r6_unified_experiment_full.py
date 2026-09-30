#!/usr/bin/env python3
"""
R6 Unified GNN-MAPPO / No-GNN Ablation Experiment

Single, reproducible implementation for:
  1) MAPPO-GNN (topology-aware 2-layer GCN)
  2) MAPPO-No-GNN (same architecture, node-wise MLP encoder)

The two scenarios share exactly the same:
  - environment and topology generation
  - SFC/request generation
  - node/VNF features
  - demand-priority execution order
  - autoregressive residual CPU/RAM reconstruction
  - feasibility masking
  - reward function
  - PPO/GAE hyperparameters
  - checkpoint selection rule
  - deterministic evaluation protocol
  - metric definitions

Only the encoder is changed: GCN vs topology-free MLP.

Recommended final protocol:
  train: seeds 11,22,33,44,55; 800 episodes
  validation: 30 requests every 50 episodes
  test: 600 requests per seed
  evaluation seed: 200000 + training_seed

This script intentionally uses the R6 placement-level reward from v754_env.py
rather than mixing it with the environment's legacy multi-agent step() reward.
Both GNN and No-GNN therefore receive exactly the same reward semantics.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import ttest_rel, wilcoxon
from statsmodels.stats.multitest import multipletests
from torch.distributions import Categorical

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    nodes: int = 20
    vnfs: int = 5
    requests: int = 20
    lifetime: int = 6

    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 3e-4
    ppo_epochs: int = 4
    clip: float = 0.20
    value_coef: float = 0.50
    entropy_coef: float = 0.01
    grad_clip: float = 0.50

    hidden_dim: int = 96
    actor_h1: int = 128
    actor_h2: int = 64
    critic_h1: int = 256
    critic_h2: int = 128

    train_episodes: int = 800
    checkpoint_every: int = 50
    validation_requests: int = 30
    test_requests: int = 600

    validation_seed_offset: int = 100000
    test_seed_offset: int = 200000

    seeds: Tuple[int, ...] = (11, 22, 33, 44, 55)

    # R6 reward used by both training scenarios.
    success_bonus: float = 10.0
    failure_penalty: float = 20.0
    violation_penalty: float = 8.0
    energy_reward_weight: float = 0.40
    latency_reward_weight: float = 0.30
    success_energy_weight: float = 0.75
    success_latency_weight: float = 1.00

    @property
    def action_space_joint(self) -> int:
        return self.nodes ** self.vnfs


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------------------------------------------------------
# Environment: unified R6 physical model
# -----------------------------------------------------------------------------

class VNFFogEnv:
    """Research-grade Fog/VNF placement environment used by both scenarios."""

    def __init__(
    self,
    cfg: Config,
    seed: int | None = None,
    requests_per_episode: int | None = None,
):
        self.cfg = cfg
        self.num_nodes = cfg.nodes
        self.num_vnfs_per_sfc = cfg.vnfs
        # self.requests_per_episode = cfg.requests
        self.requests_per_episode = (
            cfg.requests
            if requests_per_episode is None
            else int(requests_per_episode)
        )
        self.service_lifetime = cfg.lifetime
        self.rng = np.random.default_rng(seed)

        self.node_capacity = 1.0
        self.link_capacity_mbps = (500.0, 1000.0)
        self.static_power_w = 2.0
        self.dynamic_power_w = 10.0
        self.time_step_s = 0.1
        self.processing_base_ms = 0.8
        self.processing_scale_ms = 2.0
        self.edge_prop_delay_ms = (0.15, 0.8)
        self.packet_size_mb = 0.08

        self.success_bonus = cfg.success_bonus
        self.failure_penalty = cfg.failure_penalty
        self.violation_penalty = cfg.violation_penalty
        self.energy_reward_weight = cfg.energy_reward_weight
        self.latency_reward_weight = cfg.latency_reward_weight
        self.success_energy_weight = cfg.success_energy_weight
        self.success_latency_weight = cfg.success_latency_weight

        self._build_graph(0)
        self.reset(seed=seed)

    def _build_graph(self, seed: int | None) -> None:
        import networkx as nx
        m = min(2, self.num_nodes - 1)
        self.G = nx.barabasi_albert_graph(self.num_nodes, m, seed=seed)
        for i in self.G.nodes:
            self.G.nodes[i].update(cpu=1.0, ram=1.0)
        for u, v in self.G.edges:
            max_bw = float(self.rng.uniform(*self.link_capacity_mbps))
            self.G.edges[u, v].update(
                bandwidth=max_bw,
                max_bandwidth=max_bw,
                prop_delay_ms=float(self.rng.uniform(*self.edge_prop_delay_ms)),
            )

    def _generate_sfc_request(self) -> dict:
        vnfs = []
        for _ in range(self.num_vnfs_per_sfc):
            vnfs.append({
                "cpu_req": float(self.rng.uniform(0.18, 0.38)),
                "ram_req": float(self.rng.uniform(0.18, 0.38)),
            })
        return {
            "vnfs": vnfs,
            "bandwidth_req": float(self.rng.uniform(60.0, 180.0)),
            "max_latency": float(self.rng.uniform(35.0, 60.0)),
            "source_node": int(self.rng.integers(0, self.num_nodes)),
        }

    def _start_sfc(self) -> None:
        self.current_sfc_request = self._generate_sfc_request()
        self.current_vnf_index = 0
        self.current_placements: List[int] = []
        self.current_node_alloc: Dict[int, Tuple[float, float]] = {}
        self.current_link_alloc: Dict[Tuple[int, int], float] = {}
        self.current_latency_ms = 0.0
        self.current_sfc_energy = 0.0

    def _release_expired(self) -> None:
        remaining = []
        for svc in self.active_services:
            if svc["expires_at"] <= self.request_index:
                for n, (cpu, ram) in svc["node_alloc"].items():
                    self.G.nodes[n]["cpu"] = min(1.0, self.G.nodes[n]["cpu"] + cpu)
                    self.G.nodes[n]["ram"] = min(1.0, self.G.nodes[n]["ram"] + ram)
                for edge, bw in svc["link_alloc"].items():
                    self.G.edges[edge]["bandwidth"] = min(
                        self.G.edges[edge]["max_bandwidth"],
                        self.G.edges[edge]["bandwidth"] + bw,
                    )
            else:
                remaining.append(svc)
        self.active_services = remaining

    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        graph_seed = int(self.rng.integers(0, 2**31 - 1))
        self._build_graph(graph_seed)
        self.request_index = 0
        self.total_energy = 0.0
        self.accepted_requests = 0
        self.rejected_requests = 0
        self.active_services = []
        self._start_sfc()
        return self.get_node_features()

    def get_node_features(self) -> np.ndarray:
        req = self.current_sfc_request
        remaining = req["vnfs"][self.current_vnf_index:]
        total_cpu = max(1e-9, sum(v["cpu_req"] for v in req["vnfs"]))
        total_ram = max(1e-9, sum(v["ram_req"] for v in req["vnfs"]))
        global_cpu_free = np.mean([self.G.nodes[i]["cpu"] for i in self.G.nodes])
        global_ram_free = np.mean([self.G.nodes[i]["ram"] for i in self.G.nodes])
        active_ratio = min(1.0, len(self.active_services) / max(1, self.requests_per_episode))
        progress = self.current_vnf_index / max(1, self.num_vnfs_per_sfc - 1)
        x = []
        for i in range(self.num_nodes):
            degree = self.G.degree[i] / max(1, self.num_nodes - 1)
            bw_free = (
                np.mean([
                    self.G.edges[i, j]["bandwidth"] / self.G.edges[i, j]["max_bandwidth"]
                    for j in self.G.neighbors(i)
                ]) if self.G.degree[i] else 0.0
            )
            x.append([
                self.G.nodes[i]["cpu"],
                self.G.nodes[i]["ram"],
                1.0 - self.G.nodes[i]["cpu"],
                1.0 - self.G.nodes[i]["ram"],
                degree,
                bw_free,
                global_cpu_free,
                global_ram_free,
                active_ratio,
                progress,
            ])
        return np.asarray(x, dtype=np.float32)

    def get_vnf_features_all(self) -> np.ndarray:
        req = self.current_sfc_request
        total_cpu_capacity = max(1.0, float(self.num_nodes))
        total_ram_capacity = max(1.0, float(self.num_nodes))
        out = []
        for idx, v in enumerate(req["vnfs"]):
            remaining = req["vnfs"][idx:]
            out.append([
                v["cpu_req"],
                v["ram_req"],
                req["bandwidth_req"] / 200.0,
                req["max_latency"] / 60.0,
                idx / max(1, self.num_vnfs_per_sfc - 1),
                len(remaining) / max(1, self.num_vnfs_per_sfc),
                min(1.0, sum(z["cpu_req"] for z in remaining) / total_cpu_capacity),
                min(1.0, sum(z["ram_req"] for z in remaining) / total_ram_capacity),
                min(1.0, self.current_latency_ms / max(1.0, req["max_latency"])),
            ])
        return np.asarray(out, dtype=np.float32)

    def _path_info(self, src: int, dst: int):
        import networkx as nx
        path = nx.shortest_path(self.G, src, dst)
        edges = [(min(a, b), max(a, b)) for a, b in zip(path[:-1], path[1:])]
        return path, edges

    def _processing_delay(self, node: int) -> float:
        util = 1.0 - self.G.nodes[node]["cpu"]
        return self.processing_base_ms + self.processing_scale_ms * util

    @staticmethod
    def _power_from_util(util: float, static_power_w: float, dynamic_power_w: float) -> float:
        util = float(np.clip(util, 0.0, 1.0))
        active = 1.0 if util > 1e-9 else 0.0
        return active * static_power_w + dynamic_power_w * (util ** 3)

    def _network_energy_step(self) -> float:
        power = 0.0
        for i in self.G.nodes:
            util = 1.0 - self.G.nodes[i]["cpu"]
            power += self._power_from_util(util, self.static_power_w, self.dynamic_power_w)
        return power * self.time_step_s

    def _energy_reference(self) -> float:
        return max(
            1.0,
            self.num_nodes * (self.static_power_w + self.dynamic_power_w) * self.time_step_s,
        )

    def placement_metrics(self, placement: List[int]):
        """Common physical accounting for training diagnostics and evaluation."""
        import networkx as nx
        if len(placement) != self.num_vnfs_per_sfc:
            return 0.0, 0.0, False, 1.0

        cpu = {i: float(self.G.nodes[i]["cpu"]) for i in self.G.nodes}
        ram = {i: float(self.G.nodes[i]["ram"]) for i in self.G.nodes}
        bw = {e: float(self.G.edges[e]["bandwidth"]) for e in self.G.edges}
        req = self.current_sfc_request
        latency = 0.0
        energy = 0.0
        prev = req["source_node"]
        violation = 0.0

        for idx, node in enumerate(placement):
            node = int(node)
            if node not in cpu:
                violation += 1.0
                continue
            v = req["vnfs"][idx]
            cpu_before = cpu[node]
            ram_before = ram[node]
            if cpu_before + 1e-12 < v["cpu_req"]:
                violation += max(0.0, v["cpu_req"] - cpu_before)
            if ram_before + 1e-12 < v["ram_req"]:
                violation += max(0.0, v["ram_req"] - ram_before)

            cpu[node] = max(0.0, cpu_before - v["cpu_req"])
            ram[node] = max(0.0, ram_before - v["ram_req"])
            util = 1.0 - cpu[node]
            try:
                _, edges = self._path_info(prev, node)
            except nx.NetworkXNoPath:
                violation += 1.0
                edges = []

            for e in edges:
                available_bw = bw[e]
                if available_bw + 1e-12 < req["bandwidth_req"]:
                    violation += max(0.0, req["bandwidth_req"] - available_bw) / max(1.0, req["bandwidth_req"])
                bw[e] = max(0.0, available_bw - req["bandwidth_req"])
                edge = self.G.edges[e]
                latency += (
                    self.packet_size_mb * 8.0 * 1000.0 / edge["max_bandwidth"]
                    + edge["prop_delay_ms"]
                )

            latency += self.processing_base_ms + self.processing_scale_ms * util
            power = sum(
                self._power_from_util(1.0 - cpu[i], self.static_power_w, self.dynamic_power_w)
                for i in cpu
            )
            energy += power * self.time_step_s
            prev = node

        if latency > req["max_latency"]:
            violation += (latency - req["max_latency"]) / max(1.0, req["max_latency"])

        feasible = violation <= 1e-12
        return float(energy), float(latency), bool(feasible), float(violation)

    def candidate_mask(self) -> np.ndarray:
        import networkx as nx
        mask = np.zeros(self.num_nodes, dtype=bool)
        req = self.current_sfc_request
        v = req["vnfs"][self.current_vnf_index]
        src = req["source_node"] if not self.current_placements else self.current_placements[-1]
        remaining_deadline = req["max_latency"] - self.current_latency_ms
        for n in range(self.num_nodes):
            if self.G.nodes[n]["cpu"] + 1e-9 < v["cpu_req"]:
                continue
            if self.G.nodes[n]["ram"] + 1e-9 < v["ram_req"]:
                continue
            try:
                _, edges = self._path_info(src, n)
            except nx.NetworkXNoPath:
                continue
            if any(self.G.edges[e]["bandwidth"] + 1e-9 < req["bandwidth_req"] for e in edges):
                continue
            util_after = 1.0 - (self.G.nodes[n]["cpu"] - v["cpu_req"])
            proc = self.processing_base_ms + self.processing_scale_ms * util_after
            trans = sum(
                self.packet_size_mb * 8.0 * 1000.0 / self.G.edges[e]["max_bandwidth"]
                + self.G.edges[e]["prop_delay_ms"] for e in edges
            )
            if proc + trans <= remaining_deadline + 1e-9:
                mask[n] = True
        return mask

    def _simulate_candidate_state(self, node: int):
        import networkx as nx
        req = self.current_sfc_request
        v = req["vnfs"][self.current_vnf_index]
        src = req["source_node"] if not self.current_placements else self.current_placements[-1]
        try:
            _, edges = self._path_info(src, node)
        except nx.NetworkXNoPath:
            return None
        if self.G.nodes[node]["cpu"] + 1e-9 < v["cpu_req"] or self.G.nodes[node]["ram"] + 1e-9 < v["ram_req"]:
            return None
        bw = {e: float(self.G.edges[e]["bandwidth"]) for e in self.G.edges}
        if any(bw[e] + 1e-9 < req["bandwidth_req"] for e in edges):
            return None
        cpu = {i: float(self.G.nodes[i]["cpu"]) for i in self.G.nodes}
        ram = {i: float(self.G.nodes[i]["ram"]) for i in self.G.nodes}
        cpu[node] -= v["cpu_req"]
        ram[node] -= v["ram_req"]
        trans = sum(
            self.packet_size_mb * 8.0 * 1000.0 / self.G.edges[e]["max_bandwidth"]
            + self.G.edges[e]["prop_delay_ms"] for e in edges
        )
        util = 1.0 - cpu[node]
        inc = self.processing_base_ms + self.processing_scale_ms * util + trans
        for e in edges:
            bw[e] -= req["bandwidth_req"]
        return cpu, ram, bw, node, self.current_latency_ms + inc

    def _future_continuation_score(self, node: int) -> float:
        sim = self._simulate_candidate_state(node)
        if sim is None:
            return 0.0
        cpu, ram, bw, prev, latency = sim
        req = self.current_sfc_request
        remaining = req["vnfs"][self.current_vnf_index + 1:]
        if not remaining:
            return 1.0 if latency <= req["max_latency"] + 1e-9 else 0.0
        rem_cpu = sum(v["cpu_req"] for v in remaining)
        rem_ram = sum(v["ram_req"] for v in remaining)
        if sum(cpu.values()) + 1e-9 < rem_cpu or sum(ram.values()) + 1e-9 < rem_ram:
            return 0.0
        score = 1.0
        probe_prev = prev
        probe_latency = latency
        for v in remaining:
            best = None
            for dst in range(self.num_nodes):
                if cpu[dst] + 1e-9 < v["cpu_req"] or ram[dst] + 1e-9 < v["ram_req"]:
                    continue
                try:
                    _, edges = self._path_info(probe_prev, dst)
                except Exception:
                    continue
                if any(bw[e] + 1e-9 < req["bandwidth_req"] for e in edges):
                    continue
                trans = sum(
                    self.packet_size_mb * 8.0 * 1000.0 / self.G.edges[e]["max_bandwidth"]
                    + self.G.edges[e]["prop_delay_ms"] for e in edges
                )
                util = 1.0 - (cpu[dst] - v["cpu_req"])
                inc = self.processing_base_ms + self.processing_scale_ms * util + trans
                if probe_latency + inc <= req["max_latency"] + 1e-9:
                    cost = inc + 0.2 * (1.0 - cpu[dst])
                    if best is None or cost < best[0]:
                        best = (cost, dst, edges, inc)
            if best is None:
                return 0.0
            _, dst, edges, inc = best
            cpu[dst] -= v["cpu_req"]
            ram[dst] -= v["ram_req"]
            for e in edges:
                bw[e] -= req["bandwidth_req"]
            probe_prev = dst
            probe_latency += inc
        slack = max(0.0, req["max_latency"] - probe_latency)
        return float(np.clip(0.5 + 0.5 * slack / max(1.0, req["max_latency"]), 0.0, 1.0))

    def future_aware_candidate_mask(self) -> np.ndarray:
        immediate = self.candidate_mask()
        if not immediate.any():
            return immediate
        scores = np.zeros(self.num_nodes, dtype=np.float32)
        for n in np.flatnonzero(immediate):
            scores[int(n)] = self._future_continuation_score(int(n))
        future_valid = scores > 0.0
        return future_valid if future_valid.any() else immediate

    def _rollback_current_sfc(self) -> None:
        for n, (cpu, ram) in self.current_node_alloc.items():
            self.G.nodes[n]["cpu"] = min(1.0, self.G.nodes[n]["cpu"] + cpu)
            self.G.nodes[n]["ram"] = min(1.0, self.G.nodes[n]["ram"] + ram)
        for e, bw in self.current_link_alloc.items():
            self.G.edges[e]["bandwidth"] = min(
                self.G.edges[e]["max_bandwidth"],
                self.G.edges[e]["bandwidth"] + bw,
            )
        self.current_placements = []
        self.current_node_alloc = {}
        self.current_link_alloc = {}
        self.current_latency_ms = 0.0
        self.current_sfc_energy = 0.0

    def _reserve(self, node: int):
        import networkx as nx
        req = self.current_sfc_request
        v = req["vnfs"][self.current_vnf_index]
        src = req["source_node"] if not self.current_placements else self.current_placements[-1]
        path, edges = self._path_info(src, node)
        bw_req = req["bandwidth_req"]
        if self.G.nodes[node]["cpu"] + 1e-12 < v["cpu_req"] or self.G.nodes[node]["ram"] + 1e-12 < v["ram_req"]:
            return False, 0.0, 0.0
        if any(self.G.edges[e]["bandwidth"] + 1e-12 < bw_req for e in edges):
            return False, 0.0, 0.0
        old_cpu = self.G.nodes[node]["cpu"]
        self.G.nodes[node]["cpu"] -= v["cpu_req"]
        self.G.nodes[node]["ram"] -= v["ram_req"]
        self.current_node_alloc[node] = (
            self.current_node_alloc.get(node, (0.0, 0.0))[0] + v["cpu_req"],
            self.current_node_alloc.get(node, (0.0, 0.0))[1] + v["ram_req"],
        )
        trans = 0.0
        for e in edges:
            self.G.edges[e]["bandwidth"] -= bw_req
            self.current_link_alloc[e] = self.current_link_alloc.get(e, 0.0) + bw_req
            edge = self.G.edges[e]
            trans += self.packet_size_mb * 8.0 * 1000.0 / edge["max_bandwidth"] + edge["prop_delay_ms"]
        proc = self.processing_base_ms + self.processing_scale_ms * (1.0 - self.G.nodes[node]["cpu"])
        inc = proc + trans
        self.current_latency_ms += inc
        self.current_placements.append(node)
        return True, float(inc), float(old_cpu - self.G.nodes[node]["cpu"])

    def _finish_sfc(self, success: bool) -> bool:
        if success:
            self.accepted_requests += 1
            self.active_services.append({
                "expires_at": self.request_index + self.service_lifetime,
                "node_alloc": copy.deepcopy(self.current_node_alloc),
                "link_alloc": copy.deepcopy(self.current_link_alloc),
            })
        else:
            self.rejected_requests += 1
            self._rollback_current_sfc()
        self.request_index += 1
        done = self.request_index >= self.requests_per_episode
        if not done:
            self._release_expired()
            self._start_sfc()
        return done

    def commit_full_placement(self, placement: List[int]):
        """Commit a complete placement; transition semantics shared by both models."""
        _, _, feasible, _ = self.placement_metrics(placement)
        if not feasible:
            energy = self._network_energy_step()
            self.total_energy += energy
            done = self._finish_sfc(False)
            return done, False, float(energy), 0.0
        committed_energy = 0.0
        for idx, node in enumerate(placement):
            ok, _, _ = self._reserve(int(node))
            if not ok:
                self._rollback_current_sfc()
                done = self._finish_sfc(False)
                energy = self._network_energy_step()
                self.total_energy += energy
                return done, False, float(energy), 0.0
            committed_energy += self._network_energy_step()
            if idx + 1 < self.num_vnfs_per_sfc:
                self.current_vnf_index += 1
        final_latency = float(self.current_latency_ms)
        success = final_latency <= self.current_sfc_request["max_latency"] + 1e-9
        self.total_energy += committed_energy
        if not success:
            self._rollback_current_sfc()
            done = self._finish_sfc(False)
            return done, False, float(committed_energy), final_latency
        done = self._finish_sfc(True)
        return done, True, float(committed_energy), final_latency


# -----------------------------------------------------------------------------
# Shared feature/mask functions
# -----------------------------------------------------------------------------

def extract_graph_data(env: VNFFogEnv) -> GraphData:
    edges = list(env.G.edges)
    edge_index = (
        torch.tensor(edges, dtype=torch.long).t().contiguous()
        if edges else torch.empty((2, 0), dtype=torch.long)
    )
    if edge_index.numel():
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return GraphData(
        x=torch.tensor(env.get_node_features(), dtype=torch.float32),
        edge_index=edge_index,
    )


def demand_order(vnf_features_all: np.ndarray) -> List[int]:
    score = vnf_features_all[:, 0] + vnf_features_all[:, 1]
    return sorted(range(vnf_features_all.shape[0]), key=lambda i: (-float(score[i]), int(i)))


def autoregressive_state_data(data: GraphData, vnf_features_all: torch.Tensor,
                              partial_pairs: List[Tuple[int, int]]) -> Data:
    d = data.clone()
    x = d.x.clone()
    for vnf_idx, node in partial_pairs:
        x[int(node), 0] -= float(vnf_features_all[vnf_idx, 0].item())
        x[int(node), 1] -= float(vnf_features_all[vnf_idx, 1].item())
    x[:, 2] = 1.0 - x[:, 0]
    x[:, 3] = 1.0 - x[:, 1]
    d.x = x
    return d


def sequential_resource_mask(data: Data, vnf_features_all: torch.Tensor,
                             partial_pairs: List[Tuple[int, int]], vnf_idx: int):
    x = data.x.detach().cpu().numpy().astype(np.float64, copy=True)
    cpu = x[:, 0].copy()
    ram = x[:, 1].copy()
    for p_vnf, node in partial_pairs:
        cpu[int(node)] -= float(vnf_features_all[p_vnf, 0].item())
        ram[int(node)] -= float(vnf_features_all[p_vnf, 1].item())
    vcpu = float(vnf_features_all[vnf_idx, 0].item())
    vram = float(vnf_features_all[vnf_idx, 1].item())
    used = {p[0] for p in partial_pairs}
    remaining = [i for i in range(vnf_features_all.shape[0]) if i != vnf_idx and i not in used]
    rem_cpu = float(sum(vnf_features_all[i, 0].item() for i in remaining))
    rem_ram = float(sum(vnf_features_all[i, 1].item() for i in remaining))
    local = (cpu + 1e-9 >= vcpu) & (ram + 1e-9 >= vram)
    mask = local.copy()
    for node in np.where(local)[0]:
        c = cpu.copy(); r = ram.copy()
        c[node] -= vcpu; r[node] -= vram
        if c.sum() + 1e-9 < rem_cpu or r.sum() + 1e-9 < rem_ram:
            mask[node] = False
    relaxed = False
    if not mask.any():
        mask = local.copy(); relaxed = True
    if not mask.any():
        mask[:] = True; relaxed = True
    return mask, relaxed


def unified_reward(env: VNFFogEnv, energy: float, latency: float,
                   feasible: bool, violation: float) -> float:
    """Single reward used by both GNN and No-GNN training."""
    energy_norm = float(energy / env._energy_reference())
    latency_norm = float(latency / max(1.0, env.current_sfc_request["max_latency"]))
    if feasible:
        return float(
            env.success_bonus
            - env.success_energy_weight * energy_norm
            - env.success_latency_weight * latency_norm
        )
    return float(
        -env.failure_penalty
        -env.violation_penalty * min(1.0, max(0.0, float(violation)))
        -env.energy_reward_weight * energy_norm
        -env.latency_reward_weight * latency_norm
    )


# -----------------------------------------------------------------------------
# Models: identical policy except encoder
# -----------------------------------------------------------------------------

class GraphData:
    def __init__(self, x: torch.Tensor, edge_index: torch.Tensor):
        self.x = x
        self.edge_index = edge_index

    def clone(self):
        return GraphData(self.x.clone(), self.edge_index.clone())


class GCNLayer(nn.Module):
    """GCNConv-compatible graph convolution without a torch-geometric dependency."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        # Match the Glorot-style initialization used by torch-geometric GCNConv.
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.zeros_(self.lin.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        # GCNConv-style self loops.
        loop = torch.arange(n, dtype=torch.long, device=x.device)
        src = torch.cat([src, loop], dim=0)
        dst = torch.cat([dst, loop], dim=0)
        deg = torch.zeros(n, dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        inv_sqrt = deg.clamp_min(1.0).pow(-0.5)
        h = self.lin(x)
        msg = h[src] * inv_sqrt[src].unsqueeze(1) * inv_sqrt[dst].unsqueeze(1)
        out = torch.zeros_like(h)
        out.index_add_(0, dst, msg)
        return out


class GNNEncoder(nn.Module):
    def __init__(self, in_dim=10, hidden=96):
        super().__init__()
        self.conv1 = GCNLayer(in_dim, hidden)
        self.conv2 = GCNLayer(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, edge_index):
        h = self.norm1(F.relu(self.conv1(x, edge_index)))
        h = self.norm2(F.relu(self.conv2(h, edge_index)))
        return h


class NodeMLPEncoder(nn.Module):
    def __init__(self, in_dim=10, hidden=96):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, x, edge_index=None):
        h = self.norm1(F.relu(self.fc1(x)))
        h = self.norm2(F.relu(self.fc2(h)))
        return h


class UnifiedVNFMAPPO(nn.Module):
    def __init__(self, cfg: Config, encoder_type: str):
        super().__init__()
        self.cfg = cfg
        self.encoder_type = encoder_type
        if encoder_type == "gnn":
            self.encoder = GNNEncoder(10, cfg.hidden_dim)
        elif encoder_type == "mlp":
            self.encoder = NodeMLPEncoder(10, cfg.hidden_dim)
        else:
            raise ValueError("encoder_type must be 'gnn' or 'mlp'")

        actor_in = cfg.hidden_dim + 10 + 9
        self.actors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(actor_in, cfg.actor_h1), nn.ReLU(),
                nn.Linear(cfg.actor_h1, cfg.actor_h2), nn.ReLU(),
                nn.Linear(cfg.actor_h2, 1),
            ) for _ in range(cfg.vnfs)
        ])
        critic_in = cfg.hidden_dim + cfg.vnfs * 9
        self.critic = nn.Sequential(
            nn.Linear(critic_in, cfg.critic_h1), nn.ReLU(),
            nn.Linear(cfg.critic_h1, cfg.critic_h2), nn.ReLU(),
            nn.Linear(cfg.critic_h2, 1),
        )

    def encode(self, data: GraphData):
        h = self.encoder(data.x.float(), data.edge_index)
        return h, h.mean(dim=0, keepdim=True)

    def forward_agent(self, data: GraphData, vnf_feature: torch.Tensor,
                      agent_idx: int, feasible_mask=None):
        h, _ = self.encode(data)
        local = data.x.float()
        v = vnf_feature.view(1, -1).expand(self.cfg.nodes, -1)
        logits = self.actors[int(agent_idx)](torch.cat([h, local, v], dim=-1)).squeeze(-1)
        if feasible_mask is not None:
            logits = logits.masked_fill(~feasible_mask.bool(), -1e9)
        return logits

    def forward_critic(self, data: GraphData, vnf_features_all: torch.Tensor):
        _, gh = self.encode(data)
        vf = vnf_features_all.view(self.cfg.vnfs, -1)
        return self.critic(torch.cat([gh, vf.reshape(1, -1)], dim=-1))


# -----------------------------------------------------------------------------
# PPO utilities
# -----------------------------------------------------------------------------

def compute_gae(rewards, values, dones, gamma, lam):
    rewards = torch.as_tensor(rewards, dtype=torch.float32).reshape(-1)
    values = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
    dones = torch.as_tensor(dones, dtype=torch.float32).reshape(-1)
    adv = torch.zeros_like(rewards)
    gae = torch.tensor(0.0)
    for t in reversed(range(len(rewards))):
        next_v = torch.tensor(0.0) if t == len(rewards) - 1 else values[t + 1]
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    return adv, adv + values


def validation_key(summary: dict):
    return (
        float(summary["joint_feasibility"]),
        float(summary["accepted_request_rate"]),
        -float(summary["mean_latency_ms"]),
        -float(summary["mean_energy_J"]),
    )


# -----------------------------------------------------------------------------
# One complete SFC decision
# -----------------------------------------------------------------------------

def policy_decision(model: UnifiedVNFMAPPO, env: VNFFogEnv, deterministic: bool):
    data0 = extract_graph_data(env)
    vf_np = env.get_vnf_features_all()
    vf = torch.tensor(vf_np, dtype=torch.float32)
    order = demand_order(vf_np)
    partial_pairs = []
    actions = [None] * env.num_vnfs_per_sfc
    probs_by_agent = [None] * env.num_vnfs_per_sfc
    old_logps = []
    entropies = []
    relaxed_count = 0

    with torch.no_grad():
        value = model.forward_critic(data0, vf).squeeze()

    for m in order:
        data_m = autoregressive_state_data(data0, vf, partial_pairs)
        mask_np, relaxed = sequential_resource_mask(data0, vf, partial_pairs, m)
        relaxed_count += int(relaxed)
        mask = torch.tensor(mask_np, dtype=torch.bool)
        logits = model.forward_agent(data_m, vf[m], m, mask)
        dist = Categorical(logits=logits)
        action = torch.argmax(dist.probs) if deterministic else dist.sample()
        actions[m] = int(action.item())
        probs_by_agent[m] = dist.probs.detach().cpu().numpy()
        old_logps.append(dist.log_prob(action).detach())
        entropies.append(dist.entropy().detach())
        partial_pairs.append((m, int(action.item())))

    placement = [int(actions[m]) for m in range(env.num_vnfs_per_sfc)]
    return {
        "data": data0,
        "vf": vf,
        "order": order,
        "actions": actions,
        "placement": placement,
        "value": value.detach(),
        "old_logp": torch.stack(old_logps).sum().detach(),
        "entropy": torch.stack(entropies).mean().detach(),
        "probs": probs_by_agent,
        "mask_relaxed": relaxed_count,
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------

def train_one(cfg: Config, seed: int, encoder_type: str, out_dir: Path):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = VNFFogEnv(cfg, seed=seed)
    model = UnifiedVNFMAPPO(cfg, encoder_type).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    history = []
    best_key = None
    best_episode = None
    best_summary = None

    for ep in range(1, cfg.train_episodes + 1):
        env.reset(seed=seed + ep)
        traj = []
        ep_reward = 0.0
        done = False

        while not done:
            dec = policy_decision(model, env, deterministic=False)
            energy, latency, feasible, violation = env.placement_metrics(dec["placement"])
            reward = unified_reward(env, energy, latency, feasible, violation)
            done, success, _, _ = env.commit_full_placement(dec["placement"])
            ep_reward += reward
            traj.append((dec, reward, float(done)))

        rewards = torch.tensor([x[1] for x in traj], dtype=torch.float32)
        values = torch.stack([x[0]["value"] for x in traj]).float()
        dones = torch.tensor([x[2] for x in traj], dtype=torch.float32)
        adv, ret = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        oldlp = torch.stack([x[0]["old_logp"] for x in traj])

        for _ in range(cfg.ppo_epochs):
            lps, ents, vals = [], [], []
            for dec, _, _ in traj:
                data0 = dec["data"]
                vf = dec["vf"]
                order = dec["order"]
                actions = dec["actions"]
                partial_pairs = []
                lp_terms, ent_terms = [], []
                for m in order:
                    data_m = autoregressive_state_data(data0, vf, partial_pairs)
                    mask_np, _ = sequential_resource_mask(data0, vf, partial_pairs, m)
                    mask = torch.tensor(mask_np, dtype=torch.bool)
                    logits = model.forward_agent(data_m, vf[m], m, mask)
                    dist = Categorical(logits=logits)
                    a = torch.tensor(actions[m], dtype=torch.long)
                    lp_terms.append(dist.log_prob(a))
                    ent_terms.append(dist.entropy())
                    partial_pairs.append((m, actions[m]))
                lps.append(torch.stack(lp_terms).sum())
                ents.append(torch.stack(ent_terms).mean())
                vals.append(model.forward_critic(data0, vf).squeeze())

            lp = torch.stack(lps)
            ent = torch.stack(ents)
            vv = torch.stack(vals)
            ratio = torch.exp(lp - oldlp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv
            loss = (
                -torch.min(s1, s2).mean()
                + cfg.value_coef * 0.5 * (vv - ret).pow(2).mean()
                - cfg.entropy_coef * ent.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        history.append(ep_reward)
        if ep == 1 or ep % 50 == 0:
            print(
                f"[{encoder_type.upper()} seed={seed}] Episode {ep:4d} "
                f"Reward {ep_reward:9.3f} Mean50 {np.mean(history[-50:]):9.3f}"
            )

        if ep % cfg.checkpoint_every == 0:
            ckpt = out_dir / f"checkpoint_ep{ep:04d}.pt"
            torch.save(model.state_dict(), ckpt)
            val_seed = cfg.validation_seed_offset + seed
            summary, _ = evaluate_model(
                cfg, model, val_seed, cfg.validation_requests, deterministic=True
            )
            (out_dir / f"validation_ep{ep:04d}.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            key = validation_key(summary)
            if best_key is None or key > best_key:
                best_key = key
                best_episode = ep
                best_summary = summary
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                (out_dir / "selection.json").write_text(
                    json.dumps({
                        "version": "R6-unified",
                        "encoder_type": encoder_type,
                        "seed": seed,
                        "best_episode": best_episode,
                        "validation_seed": val_seed,
                        "validation_requests": cfg.validation_requests,
                        "selection_rule": "lexicographic: feasibility, acceptance, lower latency, lower energy",
                        "best_validation": best_summary,
                    }, indent=2), encoding="utf-8"
                )

    np.savetxt(
        out_dir / "training_rewards.csv",
        np.asarray(history), delimiter=",", header="reward", comments=""
    )
    return model, history, best_summary


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------

def evaluate_model(cfg: Config, model: UnifiedVNFMAPPO, eval_seed: int,
                   requests: int, deterministic: bool = True,
                   output_csv: Path | None = None):
    # env = VNFFogEnv(cfg, seed=eval_seed)
    if requests <= 0:
        raise ValueError(f"requests must be > 0, got {requests}")
    env = VNFFogEnv(
        cfg,
        seed=eval_seed,
        requests_per_episode=requests,
    )
    print(
        f"[EVAL] seed={eval_seed} "
        f"target_requests={requests} "
        f"env_requests_per_episode={env.requests_per_episode}"
    )
    model.eval()
    rows = []
    env.reset(seed=eval_seed)
    done = False
    while not done:
        dec = policy_decision(model, env, deterministic=deterministic)
        placement = dec["placement"]
        energy, latency, feasible, violation = env.placement_metrics(placement)
        request_seed_before = env.request_index
        done, success, _, _ = env.commit_full_placement(placement)
        P = np.stack(dec["probs"])
        pairwise = [
            float(np.mean(np.abs(P[i] - P[j])))
            for i in range(cfg.vnfs) for j in range(i + 1, cfg.vnfs)
        ]
        ent = [
            Categorical(probs=torch.as_tensor(P[m], dtype=torch.float32)).entropy().item()
            for m in range(cfg.vnfs)
        ]
        rows.append({
            "eval_seed": int(eval_seed),
            "request_index": int(request_seed_before),
            "placement": "-".join(map(str, placement)),
            "execution_order": "-".join(map(str, dec["order"])),
            "joint_unique_nodes": int(len(np.unique(placement))),
            "mean_agent_entropy": float(np.mean(ent)),
            "mean_pairwise_abs_prob_diff": float(np.mean(pairwise)),
            "joint_placement_feasible": bool(feasible),
            "request_success": bool(success),
            "constraint_violation": float(violation),
            "energy_J": float(energy),
            "latency_ms": float(latency),
            "mask_relaxed_agents": int(dec["mask_relaxed"]),
        })
        if len(rows) >= requests:
            break

    df = pd.DataFrame(rows)
    feasible = df["joint_placement_feasible"].astype(bool)
    success = df["request_success"].astype(bool)
    summary = {
        "eval_seed": int(eval_seed),
        "requests": int(len(df)),
        "joint_feasibility": float(feasible.mean()),
        "accepted_request_rate": float(success.mean()),
        "accepted_requests_total": int(success.sum()),
        "rejected_requests_total": int((~success).sum()),
        "joint_feasible_requests_total": int(feasible.sum()),
        "feasible_but_rejected_total": int((feasible & ~success).sum()),
        "infeasible_but_accepted_total": int((~feasible & success).sum()),
        "feasibility_acceptance_gap": float(feasible.mean() - success.mean()),
        "mean_constraint_violation": float(df["constraint_violation"].mean()),
        "mean_energy_J": float(df["energy_J"].mean()),
        "mean_latency_ms": float(df["latency_ms"].mean()),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
        "mean_unique_nodes": float(df["joint_unique_nodes"].mean()),
        "mean_pairwise_policy_diff": float(df["mean_pairwise_abs_prob_diff"].mean()),
        "mean_agent_entropy": float(df["mean_agent_entropy"].mean()),
        "mask_relaxed_decisions": int(df["mask_relaxed_agents"].sum()),
    }
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    return summary, df


def evaluate_checkpoint(cfg: Config, encoder_type: str, checkpoint: Path,
                        eval_seed: int, requests: int, output_csv: Path | None = None):
    model = UnifiedVNFMAPPO(cfg, encoder_type).cpu()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return evaluate_model(cfg, model, eval_seed, requests, deterministic=True, output_csv=output_csv)


# -----------------------------------------------------------------------------
# Five-seed experiment runner / aggregation / statistics
# -----------------------------------------------------------------------------

def run_scenario(cfg: Config, scenario: str, root: Path):
    encoder_type = "gnn" if scenario == "MAPPO-GNN" else "mlp"
    scenario_root = root / scenario
    scenario_root.mkdir(parents=True, exist_ok=True)
    all_rows = []
    summaries = []

    for seed in cfg.seeds:
        seed_root = scenario_root / f"seed{seed}"
        _, _, best = train_one(cfg, seed, encoder_type, seed_root)
        checkpoint = seed_root / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing best checkpoint: {checkpoint}")
        eval_seed = cfg.test_seed_offset + seed
        csv_path = seed_root / f"test_seed{seed}.csv"
        summary, df = evaluate_checkpoint(
            cfg, encoder_type, checkpoint, eval_seed, cfg.test_requests, csv_path
        )
        summary["method"] = scenario
        summary["source_seed"] = seed
        summary["test_seed"] = eval_seed
        summaries.append(summary)
        (seed_root / f"test_seed{seed}.summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"[{scenario} seed={seed}] TEST "
            f"feas={summary['joint_feasibility']*100:.2f}% "
            f"acc={summary['accepted_request_rate']*100:.2f}% "
            f"viol={summary['mean_constraint_violation']:.6f} "
            f"energy={summary['mean_energy_J']:.3f}J "
            f"lat={summary['mean_latency_ms']:.3f}ms"
        )

    sdf = pd.DataFrame(summaries)
    sdf.to_csv(scenario_root / "test_seed_summaries.csv", index=False)
    return sdf


def aggregate(root: Path):
    methods = ["MAPPO-GNN", "MAPPO-No-GNN"]
    frames = []
    for method in methods:
        p = root / method / "test_seed_summaries.csv"
        if not p.exists():
            raise FileNotFoundError(p)
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["method", "source_seed"]).reset_index(drop=True)
    df.to_csv(root / "ablation_seed_results.csv", index=False)

    metrics = [
        "joint_feasibility", "accepted_request_rate", "mean_constraint_violation",
        "mean_energy_J", "mean_latency_ms", "p95_latency_ms", "mean_unique_nodes",
        "mean_pairwise_policy_diff", "mask_relaxed_decisions",
    ]
    summary = df.groupby("method")[metrics].agg(["mean", "std", "count"])
    summary.to_csv(root / "ablation_method_summary.csv")

    g = df[df.method == "MAPPO-GNN"].sort_values("source_seed")
    n = df[df.method == "MAPPO-No-GNN"].sort_values("source_seed")
    if list(g.source_seed) != list(n.source_seed):
        raise RuntimeError("GNN and No-GNN seeds are not matched")

    stat_rows = []
    for metric in metrics[:-2]:
        a = g[metric].to_numpy(float)
        b = n[metric].to_numpy(float)
        t = ttest_rel(a, b)
        try:
            w = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto")
            wp = float(w.pvalue)
        except ValueError:
            wp = float("nan")
        d = a - b
        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
        stat_rows.append({
            "metric": metric,
            "n": len(a),
            "MAPPO_GNN_mean": a.mean(),
            "No_GNN_mean": b.mean(),
            "delta_GNN_minus_NoGNN": d.mean(),
            "paired_t_p": float(t.pvalue),
            "wilcoxon_p": wp,
            "cohens_dz": dz,
        })
    stats = pd.DataFrame(stat_rows)
    stats["paired_t_p_holm"] = multipletests(stats["paired_t_p"].to_numpy(), method="holm")[1]
    stats["t_significant_holm_0.05"] = stats["paired_t_p_holm"] < 0.05
    stats.to_csv(root / "ablation_paired_statistics.csv", index=False)

    return df, summary, stats



# -----------------------------------------------------------------------------
# Classical baselines and centralized single-agent PPO baseline
# -----------------------------------------------------------------------------

BASELINE_METHODS = ["Greedy", "Least-Loaded", "Random", "SA-PPO"]
ALL_METHODS = ["Greedy", "Least-Loaded", "Random", "SA-PPO", "MAPPO-GNN", "MAPPO-No-GNN"]


def _baseline_state(env: VNFFogEnv, partial_pairs: List[Tuple[int, int]], vnf_idx: int):
    """Reconstruct residual CPU/RAM/BW/latency for a partial placement."""
    import networkx as nx

    req = env.current_sfc_request
    vf = req["vnfs"]
    cpu = {i: float(env.G.nodes[i]["cpu"]) for i in env.G.nodes}
    ram = {i: float(env.G.nodes[i]["ram"]) for i in env.G.nodes}
    bw = {tuple(sorted(e)): float(env.G.edges[e]["bandwidth"]) for e in env.G.edges}
    prev = int(req["source_node"])
    latency = 0.0

    for p_vnf, node in partial_pairs:
        node = int(node)
        v = vf[int(p_vnf)]
        if cpu[node] + 1e-12 < v["cpu_req"] or ram[node] + 1e-12 < v["ram_req"]:
            return None
        try:
            _, edges = env._path_info(prev, node)
        except nx.NetworkXNoPath:
            return None
        bw_req = req["bandwidth_req"]
        if any(bw[e] + 1e-12 < bw_req for e in edges):
            return None
        cpu[node] -= v["cpu_req"]
        ram[node] -= v["ram_req"]
        for e in edges:
            bw[e] -= bw_req
            edge = env.G.edges[e]
            latency += (
                env.packet_size_mb * 8.0 * 1000.0 / edge["max_bandwidth"]
                + edge["prop_delay_ms"]
            )
        util = 1.0 - cpu[node]
        latency += env.processing_base_ms + env.processing_scale_ms * util
        prev = node

    return cpu, ram, bw, prev, latency


def _baseline_candidate_info(
    env: VNFFogEnv,
    partial_pairs: List[Tuple[int, int]],
    vnf_idx: int,
    node: int,
):
    """Return immediate-feasibility information for one baseline candidate."""
    import networkx as nx

    state = _baseline_state(env, partial_pairs, vnf_idx)
    if state is None:
        return None
    cpu, ram, bw, prev, latency = state
    req = env.current_sfc_request
    v = req["vnfs"][vnf_idx]
    node = int(node)

    if cpu[node] + 1e-12 < v["cpu_req"]:
        return None
    if ram[node] + 1e-12 < v["ram_req"]:
        return None
    try:
        _, edges = env._path_info(prev, node)
    except nx.NetworkXNoPath:
        return None

    bw_req = req["bandwidth_req"]
    if any(bw[e] + 1e-12 < bw_req for e in edges):
        return None

    cpu_after = cpu[node] - v["cpu_req"]
    ram_after = ram[node] - v["ram_req"]
    util_after = 1.0 - cpu_after
    trans = sum(
        env.packet_size_mb * 8.0 * 1000.0 / env.G.edges[e]["max_bandwidth"]
        + env.G.edges[e]["prop_delay_ms"]
        for e in edges
    )
    inc_latency = env.processing_base_ms + env.processing_scale_ms * util_after + trans
    new_latency = latency + inc_latency
    if new_latency > req["max_latency"] + 1e-9:
        return None

    return {
        "node": node,
        "cpu_after": float(cpu_after),
        "ram_after": float(ram_after),
        "util_after": float(util_after),
        "inc_latency": float(inc_latency),
        "new_latency": float(new_latency),
        "hops": int(len(edges)),
        "degree": int(env.G.degree[node]),
    }


def baseline_policy_decision(env: VNFFogEnv, method: str, rng: np.random.Generator):
    """Construct a complete SFC placement for a classical baseline."""
    vf_np = env.get_vnf_features_all()
    order = demand_order(vf_np)
    partial_pairs: List[Tuple[int, int]] = []
    placement = [None] * env.num_vnfs_per_sfc
    relaxed_count = 0

    for m in order:
        data0 = extract_graph_data(env)
        vf = torch.tensor(vf_np, dtype=torch.float32)
        mask_np, relaxed = sequential_resource_mask(data0, vf, partial_pairs, m)
        relaxed_count += int(relaxed)

        candidates = []
        for node in np.flatnonzero(mask_np):
            info = _baseline_candidate_info(env, partial_pairs, m, int(node))
            if info is not None:
                candidates.append(info)

        # If the common predictive mask is too restrictive, use the immediate
        # physical-feasibility set. This keeps all baselines operational without
        # modifying the environment's final commit semantics.
        if not candidates:
            for node in range(env.num_nodes):
                info = _baseline_candidate_info(env, partial_pairs, m, int(node))
                if info is not None:
                    candidates.append(info)

        if not candidates:
            # No physically valid next action. Return a deterministic placeholder;
            # commit_full_placement will reject it and preserve accounting semantics.
            placement[m] = 0
            partial_pairs.append((m, 0))
            continue

        if method == "Random":
            chosen = candidates[int(rng.integers(0, len(candidates)))]
        elif method == "Least-Loaded":
            # Maximize residual CPU/RAM jointly; use incremental latency as tie-break.
            chosen = min(
                candidates,
                key=lambda z: (
                    -0.5 * (z["cpu_after"] + z["ram_after"]),
                    z["inc_latency"],
                    -z["degree"],
                    z["node"],
                ),
            )
        elif method == "Greedy":
            # Minimize immediate latency first, then prefer a less-loaded node.
            chosen = min(
                candidates,
                key=lambda z: (
                    z["inc_latency"],
                    1.0 - 0.5 * (z["cpu_after"] + z["ram_after"]),
                    z["hops"],
                    z["node"],
                ),
            )
        else:
            raise ValueError(f"Unknown classical baseline: {method}")

        placement[m] = int(chosen["node"])
        partial_pairs.append((m, int(chosen["node"])))

    return {
        "placement": [int(x) for x in placement],
        "order": order,
        "mask_relaxed": relaxed_count,
    }


def evaluate_classical_baseline(
    cfg: Config,
    method: str,
    eval_seed: int,
    requests: int,
    output_csv: Path | None = None,
):
    if method not in ["Greedy", "Least-Loaded", "Random"]:
        raise ValueError(method)
    if requests <= 0:
        raise ValueError(f"requests must be > 0, got {requests}")

    rng = np.random.default_rng(eval_seed)
    env = VNFFogEnv(cfg, seed=eval_seed, requests_per_episode=requests)
    env.reset(seed=eval_seed)
    rows = []
    done = False

    while not done and len(rows) < requests:
        dec = baseline_policy_decision(env, method, rng)
        placement = dec["placement"]
        energy, latency, feasible, violation = env.placement_metrics(placement)
        request_index = env.request_index
        done, success, _, _ = env.commit_full_placement(placement)
        rows.append({
            "eval_seed": int(eval_seed),
            "request_index": int(request_index),
            "method": method,
            "placement": "-".join(map(str, placement)),
            "execution_order": "-".join(map(str, dec["order"])),
            "joint_unique_nodes": int(len(np.unique(placement))),
            "mean_agent_entropy": np.nan,
            "mean_pairwise_abs_prob_diff": np.nan,
            "joint_placement_feasible": bool(feasible),
            "request_success": bool(success),
            "constraint_violation": float(violation),
            "energy_J": float(energy),
            "latency_ms": float(latency),
            "mask_relaxed_agents": int(dec["mask_relaxed"]),
        })

    df = pd.DataFrame(rows)
    if len(df) != requests:
        raise RuntimeError(f"{method}: expected {requests} requests, got {len(df)}")

    feasible = df["joint_placement_feasible"].astype(bool)
    success = df["request_success"].astype(bool)
    summary = {
        "eval_seed": int(eval_seed),
        "requests": int(len(df)),
        "joint_feasibility": float(feasible.mean()),
        "accepted_request_rate": float(success.mean()),
        "accepted_requests_total": int(success.sum()),
        "rejected_requests_total": int((~success).sum()),
        "joint_feasible_requests_total": int(feasible.sum()),
        "feasible_but_rejected_total": int((feasible & ~success).sum()),
        "infeasible_but_accepted_total": int((~feasible & success).sum()),
        "feasibility_acceptance_gap": float(feasible.mean() - success.mean()),
        "mean_constraint_violation": float(df["constraint_violation"].mean()),
        "mean_energy_J": float(df["energy_J"].mean()),
        "mean_latency_ms": float(df["latency_ms"].mean()),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
        "mean_unique_nodes": float(df["joint_unique_nodes"].mean()),
        "mean_pairwise_policy_diff": np.nan,
        "mean_agent_entropy": np.nan,
        "mask_relaxed_decisions": int(df["mask_relaxed_agents"].sum()),
        "method": method,
    }
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    return summary, df


def run_classical_baselines(cfg: Config, root: Path):
    """Evaluate Greedy, Least-Loaded, and Random on the same five-seed protocol."""
    frames = []
    for method in ["Greedy", "Least-Loaded", "Random"]:
        method_root = root / method
        method_root.mkdir(parents=True, exist_ok=True)
        summaries = []
        for seed in cfg.seeds:
            eval_seed = cfg.test_seed_offset + seed
            csv_path = method_root / f"test_seed{seed}.csv"
            summary, _ = evaluate_classical_baseline(
                cfg, method, eval_seed, cfg.test_requests, csv_path
            )
            summary["source_seed"] = seed
            summary["test_seed"] = eval_seed
            summaries.append(summary)
            (method_root / f"test_seed{seed}.summary.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            print(
                f"[{method} seed={seed}] TEST "
                f"feas={summary['joint_feasibility']*100:.2f}% "
                f"acc={summary['accepted_request_rate']*100:.2f}% "
                f"viol={summary['mean_constraint_violation']:.6f} "
                f"energy={summary['mean_energy_J']:.3f}J "
                f"lat={summary['mean_latency_ms']:.3f}ms"
            )
        sdf = pd.DataFrame(summaries)
        sdf.to_csv(method_root / "test_seed_summaries.csv", index=False)
        frames.append(sdf)
    return pd.concat(frames, ignore_index=True)


class CentralizedSAPPPO(nn.Module):
    """Centralized single-agent PPO with one shared autoregressive actor."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.encoder = GNNEncoder(10, cfg.hidden_dim)
        actor_in = cfg.hidden_dim + 10 + 9
        self.actor = nn.Sequential(
            nn.Linear(actor_in, cfg.actor_h1), nn.ReLU(),
            nn.Linear(cfg.actor_h1, cfg.actor_h2), nn.ReLU(),
            nn.Linear(cfg.actor_h2, 1),
        )
        critic_in = cfg.hidden_dim + cfg.vnfs * 9
        self.critic = nn.Sequential(
            nn.Linear(critic_in, cfg.critic_h1), nn.ReLU(),
            nn.Linear(cfg.critic_h1, cfg.critic_h2), nn.ReLU(),
            nn.Linear(cfg.critic_h2, 1),
        )

    def encode(self, data: GraphData):
        h = self.encoder(data.x.float(), data.edge_index)
        return h, h.mean(dim=0, keepdim=True)

    def forward_actor(self, data: GraphData, vnf_feature: torch.Tensor, mask=None):
        h, _ = self.encode(data)
        local = data.x.float()
        v = vnf_feature.view(1, -1).expand(self.cfg.nodes, -1)
        logits = self.actor(torch.cat([h, local, v], dim=-1)).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask.bool(), -1e9)
        return logits

    def forward_critic(self, data: GraphData, vnf_features_all: torch.Tensor):
        _, gh = self.encode(data)
        return self.critic(torch.cat([gh, vnf_features_all.view(1, -1)], dim=-1))


def sappo_policy_decision(model: CentralizedSAPPPO, env: VNFFogEnv, deterministic: bool):
    data0 = extract_graph_data(env)
    vf_np = env.get_vnf_features_all()
    vf = torch.tensor(vf_np, dtype=torch.float32)
    order = demand_order(vf_np)
    partial_pairs = []
    actions = [None] * env.num_vnfs_per_sfc
    probs_by_agent = [None] * env.num_vnfs_per_sfc
    old_logps, entropies = [], []
    relaxed_count = 0

    with torch.no_grad():
        value = model.forward_critic(data0, vf).squeeze()

    for m in order:
        data_m = autoregressive_state_data(data0, vf, partial_pairs)
        mask_np, relaxed = sequential_resource_mask(data0, vf, partial_pairs, m)
        relaxed_count += int(relaxed)
        mask = torch.tensor(mask_np, dtype=torch.bool)
        logits = model.forward_actor(data_m, vf[m], mask)
        dist = Categorical(logits=logits)
        action = torch.argmax(dist.probs) if deterministic else dist.sample()
        actions[m] = int(action.item())
        probs_by_agent[m] = dist.probs.detach().cpu().numpy()
        old_logps.append(dist.log_prob(action).detach())
        entropies.append(dist.entropy().detach())
        partial_pairs.append((m, int(action.item())))

    return {
        "data": data0,
        "vf": vf,
        "order": order,
        "actions": actions,
        "placement": [int(actions[m]) for m in range(env.num_vnfs_per_sfc)],
        "value": value.detach(),
        "old_logp": torch.stack(old_logps).sum().detach(),
        "entropy": torch.stack(entropies).mean().detach(),
        "probs": probs_by_agent,
        "mask_relaxed": relaxed_count,
    }


def train_sappo_one(cfg: Config, seed: int, out_dir: Path):
    set_seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = VNFFogEnv(cfg, seed=seed)
    model = CentralizedSAPPPO(cfg).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history = []
    best_key = None
    best_episode = None
    best_summary = None

    for ep in range(1, cfg.train_episodes + 1):
        env.reset(seed=seed + ep)
        traj = []
        ep_reward = 0.0
        done = False
        while not done:
            dec = sappo_policy_decision(model, env, deterministic=False)
            energy, latency, feasible, violation = env.placement_metrics(dec["placement"])
            reward = unified_reward(env, energy, latency, feasible, violation)
            done, _, _, _ = env.commit_full_placement(dec["placement"])
            ep_reward += reward
            traj.append((dec, reward, float(done)))

        rewards = torch.tensor([x[1] for x in traj], dtype=torch.float32)
        values = torch.stack([x[0]["value"] for x in traj]).float()
        dones = torch.tensor([x[2] for x in traj], dtype=torch.float32)
        adv, ret = compute_gae(rewards, values, dones, cfg.gamma, cfg.gae_lambda)
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
        oldlp = torch.stack([x[0]["old_logp"] for x in traj])

        for _ in range(cfg.ppo_epochs):
            lps, ents, vals = [], [], []
            for dec, _, _ in traj:
                data0, vf, order, actions = dec["data"], dec["vf"], dec["order"], dec["actions"]
                partial_pairs = []
                lp_terms, ent_terms = [], []
                for m in order:
                    data_m = autoregressive_state_data(data0, vf, partial_pairs)
                    mask_np, _ = sequential_resource_mask(data0, vf, partial_pairs, m)
                    logits = model.forward_actor(data_m, vf[m], torch.tensor(mask_np, dtype=torch.bool))
                    dist = Categorical(logits=logits)
                    a = torch.tensor(actions[m], dtype=torch.long)
                    lp_terms.append(dist.log_prob(a))
                    ent_terms.append(dist.entropy())
                    partial_pairs.append((m, actions[m]))
                lps.append(torch.stack(lp_terms).sum())
                ents.append(torch.stack(ent_terms).mean())
                vals.append(model.forward_critic(data0, vf).squeeze())

            lp, ent, vv = torch.stack(lps), torch.stack(ents), torch.stack(vals)
            ratio = torch.exp(lp - oldlp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * adv
            loss = (
                -torch.min(s1, s2).mean()
                + cfg.value_coef * 0.5 * (vv - ret).pow(2).mean()
                - cfg.entropy_coef * ent.mean()
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

        history.append(ep_reward)
        if ep == 1 or ep % 50 == 0:
            print(
                f"[SA-PPO seed={seed}] Episode {ep:4d} "
                f"Reward {ep_reward:9.3f} Mean50 {np.mean(history[-50:]):9.3f}"
            )

        if ep % cfg.checkpoint_every == 0:
            ckpt = out_dir / f"checkpoint_ep{ep:04d}.pt"
            torch.save(model.state_dict(), ckpt)
            val_seed = cfg.validation_seed_offset + seed
            summary, _ = evaluate_sappo_model(cfg, model, val_seed, cfg.validation_requests)
            (out_dir / f"validation_ep{ep:04d}.json").write_text(
                json.dumps(summary, indent=2), encoding="utf-8"
            )
            key = validation_key(summary)
            if best_key is None or key > best_key:
                best_key = key
                best_episode = ep
                best_summary = summary
                torch.save(model.state_dict(), out_dir / "best_model.pt")
                (out_dir / "selection.json").write_text(
                    json.dumps({
                        "version": "R6-unified-SAPPO",
                        "method": "SA-PPO",
                        "seed": seed,
                        "best_episode": best_episode,
                        "validation_seed": val_seed,
                        "validation_requests": cfg.validation_requests,
                        "selection_rule": "lexicographic: feasibility, acceptance, lower latency, lower energy",
                        "best_validation": best_summary,
                    }, indent=2), encoding="utf-8"
                )

    np.savetxt(
        out_dir / "training_rewards.csv",
        np.asarray(history), delimiter=",", header="reward", comments=""
    )
    return model, history, best_summary


def evaluate_sappo_model(
    cfg: Config,
    model: CentralizedSAPPPO,
    eval_seed: int,
    requests: int,
    output_csv: Path | None = None,
):
    env = VNFFogEnv(cfg, seed=eval_seed, requests_per_episode=requests)
    env.reset(seed=eval_seed)
    model.eval()
    rows = []
    done = False
    while not done and len(rows) < requests:
        dec = sappo_policy_decision(model, env, deterministic=True)
        energy, latency, feasible, violation = env.placement_metrics(dec["placement"])
        request_index = env.request_index
        done, success, _, _ = env.commit_full_placement(dec["placement"])
        P = np.stack(dec["probs"])
        pairwise = [
            float(np.mean(np.abs(P[i] - P[j])))
            for i in range(cfg.vnfs) for j in range(i + 1, cfg.vnfs)
        ]
        ent = [
            Categorical(probs=torch.as_tensor(P[m], dtype=torch.float32)).entropy().item()
            for m in range(cfg.vnfs)
        ]
        rows.append({
            "eval_seed": int(eval_seed),
            "request_index": int(request_index),
            "method": "SA-PPO",
            "placement": "-".join(map(str, dec["placement"])),
            "execution_order": "-".join(map(str, dec["order"])),
            "joint_unique_nodes": int(len(np.unique(dec["placement"]))),
            "mean_agent_entropy": float(np.mean(ent)),
            "mean_pairwise_abs_prob_diff": float(np.mean(pairwise)),
            "joint_placement_feasible": bool(feasible),
            "request_success": bool(success),
            "constraint_violation": float(violation),
            "energy_J": float(energy),
            "latency_ms": float(latency),
            "mask_relaxed_agents": int(dec["mask_relaxed"]),
        })
    df = pd.DataFrame(rows)
    if len(df) != requests:
        raise RuntimeError(f"SA-PPO: expected {requests} requests, got {len(df)}")
    feasible = df["joint_placement_feasible"].astype(bool)
    success = df["request_success"].astype(bool)
    summary = {
        "eval_seed": int(eval_seed),
        "requests": int(len(df)),
        "joint_feasibility": float(feasible.mean()),
        "accepted_request_rate": float(success.mean()),
        "accepted_requests_total": int(success.sum()),
        "rejected_requests_total": int((~success).sum()),
        "joint_feasible_requests_total": int(feasible.sum()),
        "feasible_but_rejected_total": int((feasible & ~success).sum()),
        "infeasible_but_accepted_total": int((~feasible & success).sum()),
        "feasibility_acceptance_gap": float(feasible.mean() - success.mean()),
        "mean_constraint_violation": float(df["constraint_violation"].mean()),
        "mean_energy_J": float(df["energy_J"].mean()),
        "mean_latency_ms": float(df["latency_ms"].mean()),
        "p95_latency_ms": float(df["latency_ms"].quantile(0.95)),
        "mean_unique_nodes": float(df["joint_unique_nodes"].mean()),
        "mean_pairwise_policy_diff": float(df["mean_pairwise_abs_prob_diff"].mean()),
        "mean_agent_entropy": float(df["mean_agent_entropy"].mean()),
        "mask_relaxed_decisions": int(df["mask_relaxed_agents"].sum()),
        "method": "SA-PPO",
    }
    if output_csv is not None:
        df.to_csv(output_csv, index=False)
    return summary, df


def run_sappo(cfg: Config, root: Path):
    scenario_root = root / "SA-PPO"
    scenario_root.mkdir(parents=True, exist_ok=True)
    summaries = []
    for seed in cfg.seeds:
        seed_root = scenario_root / f"seed{seed}"
        _, _, best = train_sappo_one(cfg, seed, seed_root)
        checkpoint = seed_root / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"Missing SA-PPO best checkpoint: {checkpoint}")
        model = CentralizedSAPPPO(cfg).cpu()
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        eval_seed = cfg.test_seed_offset + seed
        csv_path = seed_root / f"test_seed{seed}.csv"
        summary, _ = evaluate_sappo_model(cfg, model, eval_seed, cfg.test_requests, csv_path)
        summary["source_seed"] = seed
        summary["test_seed"] = eval_seed
        summaries.append(summary)
        (seed_root / f"test_seed{seed}.summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        print(
            f"[SA-PPO seed={seed}] TEST "
            f"feas={summary['joint_feasibility']*100:.2f}% "
            f"acc={summary['accepted_request_rate']*100:.2f}% "
            f"viol={summary['mean_constraint_violation']:.6f} "
            f"energy={summary['mean_energy_J']:.3f}J "
            f"lat={summary['mean_latency_ms']:.3f}ms"
        )
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(scenario_root / "test_seed_summaries.csv", index=False)
    return sdf


def aggregate_full_benchmark(root: Path):
    methods = ["Greedy", "Least-Loaded", "Random", "SA-PPO", "MAPPO-GNN", "MAPPO-No-GNN"]
    frames = []
    for method in methods:
        p = root / method / "test_seed_summaries.csv"
        if not p.exists():
            raise FileNotFoundError(f"Missing benchmark summary: {p}")
        frames.append(pd.read_csv(p))
    df = pd.concat(frames, ignore_index=True)
    df = df.sort_values(["method", "source_seed"]).reset_index(drop=True)
    df.to_csv(root / "benchmark_seed_results.csv", index=False)

    metrics = [
        "joint_feasibility", "accepted_request_rate", "mean_constraint_violation",
        "mean_energy_J", "mean_latency_ms", "p95_latency_ms", "mean_unique_nodes",
        "mean_pairwise_policy_diff", "mean_agent_entropy", "mask_relaxed_decisions",
    ]
    summary = df.groupby("method")[metrics].agg(["mean", "std", "count"])
    summary.to_csv(root / "benchmark_method_summary.csv")

    # Preserve the original paired GNN-vs-No-GNN analysis.
    g = df[df.method == "MAPPO-GNN"].sort_values("source_seed")
    n = df[df.method == "MAPPO-No-GNN"].sort_values("source_seed")
    if list(g.source_seed) != list(n.source_seed):
        raise RuntimeError("GNN and No-GNN seeds are not matched")
    stat_rows = []
    for metric in [
        "joint_feasibility", "accepted_request_rate", "mean_constraint_violation",
        "mean_energy_J", "mean_latency_ms", "p95_latency_ms", "mean_unique_nodes",
    ]:
        a = g[metric].to_numpy(float); b = n[metric].to_numpy(float)
        t = ttest_rel(a, b)
        try:
            w = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto")
            wp = float(w.pvalue)
        except ValueError:
            wp = float("nan")
        d = a - b
        dz = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else float("nan")
        stat_rows.append({
            "metric": metric, "n": len(a),
            "MAPPO_GNN_mean": a.mean(), "No_GNN_mean": b.mean(),
            "delta_GNN_minus_NoGNN": d.mean(), "paired_t_p": float(t.pvalue),
            "wilcoxon_p": wp, "cohens_dz": dz,
        })
    stats = pd.DataFrame(stat_rows)
    stats["paired_t_p_holm"] = multipletests(stats["paired_t_p"].to_numpy(), method="holm")[1]
    stats["t_significant_holm_0.05"] = stats["paired_t_p_holm"] < 0.05
    stats.to_csv(root / "ablation_paired_statistics.csv", index=False)
    return df, summary, stats

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def make_cfg(args) -> Config:
    cfg = Config(
        train_episodes=args.episodes,
        validation_requests=args.validation_requests,
        test_requests=args.test_requests,
    )
    if args.seeds:
        cfg = dataclass_replace(cfg, seeds=tuple(args.seeds))
    return cfg


def dataclass_replace(obj, **kwargs):
    values = asdict(obj)
    values.update(kwargs)
    return Config(**values)


def main():
    p = argparse.ArgumentParser(
        description="R6 complete VNF placement benchmark: MAPPO-GNN, No-GNN, Greedy, Least-Loaded, Random, SA-PPO"
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default="results_r6_complete")
    common.add_argument("--episodes", type=int, default=800)
    common.add_argument("--validation-requests", type=int, default=30)
    common.add_argument("--test-requests", type=int, default=600)
    common.add_argument("--seeds", nargs="+", type=int, default=None)

    t = sub.add_parser("train-test", parents=[common])
    t.add_argument(
        "--scenario",
        choices=[
            "MAPPO-GNN", "MAPPO-No-GNN", "Greedy", "Least-Loaded", "Random",
            "SA-PPO", "baselines", "all", "both",
        ],
        default="all",
    )

    a = sub.add_parser("aggregate", parents=[common])
    s = sub.add_parser("smoke", parents=[common])

    args = p.parse_args()
    cfg = make_cfg(args)
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)
    (root / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    if args.cmd == "smoke":
        smoke_cfg = dataclass_replace(
            cfg,
            train_episodes=2,
            checkpoint_every=2,
            validation_requests=2,
            test_requests=5,
            seeds=(11,),
        )
        print("Running GNN smoke test...")
        run_scenario(smoke_cfg, "MAPPO-GNN", root / "smoke")
        print("Running No-GNN smoke test...")
        run_scenario(smoke_cfg, "MAPPO-No-GNN", root / "smoke")
        print("Running SA-PPO smoke test...")
        run_sappo(smoke_cfg, root / "smoke")
        print("Running classical baseline smoke tests...")
        run_classical_baselines(smoke_cfg, root / "smoke")
        aggregate_full_benchmark(root / "smoke")
        print("SMOKE TEST PASSED")
        return

    if args.cmd == "train-test":
        scenario = args.scenario
        if scenario == "all":
            run_scenario(cfg, "MAPPO-GNN", root)
            run_scenario(cfg, "MAPPO-No-GNN", root)
            run_sappo(cfg, root)
            run_classical_baselines(cfg, root)
            aggregate_full_benchmark(root)
        elif scenario == "both":
            run_scenario(cfg, "MAPPO-GNN", root)
            run_scenario(cfg, "MAPPO-No-GNN", root)
            aggregate(root)
        elif scenario == "baselines":
            run_sappo(cfg, root)
            run_classical_baselines(cfg, root)
            aggregate_full_benchmark(root)
        elif scenario == "SA-PPO":
            run_sappo(cfg, root)
        elif scenario in ["Greedy", "Least-Loaded", "Random"]:
            method_root = root / scenario
            method_root.mkdir(parents=True, exist_ok=True)
            frames = []
            for seed in cfg.seeds:
                eval_seed = cfg.test_seed_offset + seed
                summary, _ = evaluate_classical_baseline(
                    cfg, scenario, eval_seed, cfg.test_requests,
                    method_root / f"test_seed{seed}.csv"
                )
                summary["source_seed"] = seed
                summary["test_seed"] = eval_seed
                frames.append(summary)
            pd.DataFrame(frames).to_csv(method_root / "test_seed_summaries.csv", index=False)
        else:
            run_scenario(cfg, scenario, root)
        print(f"Results written to: {root.resolve()}")
        return

    if args.cmd == "aggregate":
        _, summary, stats = aggregate_full_benchmark(root)
        print(summary)
        print("\nPaired GNN vs No-GNN statistics:")
        print(stats.to_string(index=False))
        return


if __name__ == "__main__":
    main()
