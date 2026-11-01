def test_trusttab_package():
    import trusttab

    assert trusttab is not None
    assert trusttab.__version__ == "0.1.0"