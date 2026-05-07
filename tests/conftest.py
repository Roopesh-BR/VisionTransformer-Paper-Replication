def pytest_collection_modifyitems(session, config, items):
    """Remove test items whose actual function comes from the vit package."""
    filtered = []
    for item in items:
        fn = getattr(item, "function", None)
        module = getattr(fn, "__module__", "") if fn else ""
        if module.startswith("vit."):
            continue
        filtered.append(item)
    items[:] = filtered
