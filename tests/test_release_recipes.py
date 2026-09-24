"""Meaningful contract checks for the published recipe and conditional CE."""
import importlib.util
from pathlib import Path
import torch
from regcfpo.training.gate import compute_global_gate_payload

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("train_paper",ROOT/"scripts/train_paper.py")
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def value(cmd,key):
    return cmd[cmd.index(key)+1]

def test_published_reg_gate_is_effectively_binary_in_single_group():
    for k in range(5):
        rewards=torch.tensor([1.]*k+[0.]*(4-k))
        p=compute_global_gate_payload(rewards,4)
        assert p.normalization_applied
        expected=0. if k==0 else 1.
        assert abs(p.weights.item()-expected)<4e-6

def test_bridge_ce_uses_conditional_options_not_vocabulary_mass():
    logits=torch.tensor([[1.,2.,-1.,0.]],requires_grad=True)
    a=torch.nn.functional.cross_entropy(logits,torch.tensor([1]))
    b=torch.nn.functional.cross_entropy(logits-12.,torch.tensor([1]))
    torch.testing.assert_close(a,b)

def test_published_reg_arms_match_steps_and_optimization():
    for seed in [1234,20260910]:
        cmds=[module.command("reg",arm,seed,"initial","run") for arm in ["reg","continued","local"]]
        for cmd in cmds:
            assert value(cmd,"--max-steps")=="50"
            assert value(cmd,"--beta")=="0"
            assert value(cmd,"--gradient-accumulation-steps")=="1"
            assert value(cmd,"--model-path")=="initial"
        assert value(cmds[0],"--gamma")=="0.0176"
        assert value(cmds[2],"--gamma")=="0.0104"
        assert value(cmds[0],"--gate-credit-mode")=="legacy_closed_form_normalized"

def test_full_adds_only_bridge_weight_to_prefix_recipe():
    a=module.command("bridge","prefix",2001,"initial","run")
    b=module.command("bridge","full",2001,"initial","run")
    i=a.index("--joint-answer-bridge-weight")+1
    assert a[:i]==b[:i] and a[i+1:]==b[i+1:]
    assert a[i]=="0.0" and b[i]=="0.01"
    assert value(a,"--joint-prefix-credit-scope")=="reasoning_all_wrong"
    assert value(a,"--joint-trust-weight")=="0.1"
