import nengo


def pytest_funcarg__Simulator(request):
    """the Simulator class being tested.

    Please use this, and not nengo.Simulator directly,
    unless the test is reference simulator specific.
    """
    return nengo.Simulator


def pytest_generate_tests(metafunc):
    if "Neurons" in metafunc.funcargnames:
        metafunc.parametrize("Neurons",
                             [nengo.LIF, nengo.LIFRate, nengo.Direct])
    if "RateSpiking" in metafunc.funcargnames:
        metafunc.parametrize("RateSpiking", [nengo.LIF, nengo.LIFRate])
    if "Spiking" in metafunc.funcargnames:
        metafunc.parametrize("Spiking", [nengo.LIF])
