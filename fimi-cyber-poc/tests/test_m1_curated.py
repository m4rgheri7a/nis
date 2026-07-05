from fimicyber.config import load_config
from fimicyber.loaders.curated import load_events


def test_curated_doppelganger_loads_real_iocs():
    cfg = load_config()
    events = load_events(cfg.data_dir / "curated", cfg, [])

    assert len(events) == 6
    assert {ev.source_dataset for ev in events} == {"curated_doppelganger"}
    assert {ev.campaign_id_source for ev in events} == {"curated"}

    real_iocs = [
        ioc
        for ev in events
        for ioc in ev.iocs
        if ioc.category == "OperationalIOC" and not ioc.synthetic
    ]
    assert len(real_iocs) >= 50
    assert any(ioc.value == "bild.asia" and ioc.ioc_type == "domain" for ioc in real_iocs)
    assert any(ioc.value == "46.246.96.73" and ioc.ioc_type == "ipv4" for ioc in real_iocs)
    assert any(ioc.value == "arturo.ns.cloudflare.com" and ioc.ioc_type == "ns" for ioc in real_iocs)
