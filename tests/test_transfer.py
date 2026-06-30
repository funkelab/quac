import torch

from quac.training.stargan import build_model, transfer_to_more_domains, _unwrap


def _nets(num_domains):
    return build_model(
        img_size=64,
        style_dim=64,
        latent_dim=16,
        num_domains=num_domains,
        input_dim=1,
        variational=True,
    )


def test_transfer_to_more_domains_broadcasts_heads():
    src_nets, _ = _nets(num_domains=1)
    dst_nets, _ = _nets(num_domains=3)

    transfer_to_more_domains(src_nets, dst_nets)

    src_enc = _unwrap(src_nets["style_encoder"])
    dst_enc = _unwrap(dst_nets["style_encoder"])
    # Every target style head is initialised from the single source head.
    src_w = src_enc.unshared[0].weight
    for head in dst_enc.unshared:
        assert torch.equal(head.weight, src_w)

    # The discriminator's domain conv is tiled across the 3 output channels.
    src_d = _unwrap(src_nets["discriminator"]).main[-1]
    dst_d = _unwrap(dst_nets["discriminator"]).main[-1]
    assert dst_d.weight.size(0) == 3
    for i in range(3):
        assert torch.equal(dst_d.weight[i], src_d.weight[0])

    # The (domain-agnostic) generator is copied exactly.
    src_g = dict(_unwrap(src_nets["generator"]).named_parameters())
    for name, p in _unwrap(dst_nets["generator"]).named_parameters():
        assert torch.equal(p, src_g[name])


def test_transfer_excludes_mapping_network_by_default():
    src_nets, _ = _nets(num_domains=1)
    dst_nets, _ = _nets(num_domains=3)

    # Fresh mapping network before transfer.
    before = _unwrap(dst_nets["mapping_network"]).unshared[0][0].weight.clone()
    transfer_to_more_domains(src_nets, dst_nets)
    after = _unwrap(dst_nets["mapping_network"]).unshared[0][0].weight
    # Mapping network is untouched (introduced fresh at fine-tune).
    assert torch.equal(before, after)
