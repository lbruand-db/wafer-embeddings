from wafer_embeddings.jobs.serve import _args, desired_entity, serve_action

FQN = "mmf_mlops_demo_catalog.wafer_embeddings.wafer_encoder"


def test_desired_entity_serves_the_given_version():
    e = desired_entity(FQN, "3")
    assert e == {
        "name": "wafer-encoder",
        "entity_name": FQN,
        "entity_version": "3",
        "workload_size": "Small",
        "scale_to_zero_enabled": True,
    }


def test_serve_action_create_update_ok():
    want = desired_entity(FQN, "2")
    assert serve_action(None, want) == "create"
    assert serve_action([dict(want)], want) == "ok"  # idempotent
    assert serve_action([dict(want, entity_version="1")], want) == "update"  # new champion
    assert serve_action([dict(want, workload_size="Medium")], want) == "update"
    assert serve_action([want, dict(want, name="other")], want) == "update"  # 2 entities
    assert serve_action([], want) == "update"  # endpoint exists but serves nothing


def test_args_defaults():
    a = _args(["--catalog", "c", "--schema", "s"])
    assert (a.model, a.alias, a.endpoint) == ("wafer_encoder", "champion", "wafer-encoder")
    assert a.workload_size == "Small" and a.scale_to_zero is True
