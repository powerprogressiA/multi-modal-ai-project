from sensors.simulated_sensor import SimulatedSensor

def test_smoke_simulated_sensor_returns_frame():
    sim = SimulatedSensor()
    frame = sim.get_frame()
    assert frame is not None
