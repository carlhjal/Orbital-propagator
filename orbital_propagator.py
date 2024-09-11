"""
Program defaults to reading from a local text file.
Pass argument <fetch> to get latest data (actually any extra argument will do):
python3 orbital_propagator.py <fetch>
"""

import sys
import requests
from matplotlib import pyplot as plt
from module.satellite import *
from module.plot import *

def fetch_data():
    url = "https://celestrak.org/NORAD/elements/cubesat.txt"
    page = requests.get(url=url, timeout=3)
    return page.text.rstrip().split("\n")

def main():
    raw_data = ""

    if len(sys.argv) > 1:
        fetch = (sys.argv[1].lower() == "fetch")
        raw_data = fetch_data()
    else:
        with open("raw_data_04_06.txt", "r", encoding="utf8") as f:
            raw_data = f.read().rstrip().split("\n")
    
    # close if too many requests:
    if raw_data[0][0] == "<":
        print("error, probably too many requests")
        sys.exit()

    estcube_index = [i for i, curr_str in enumerate(raw_data) if "ESTCUBE" in curr_str][0]
    estcube_data = raw_data[estcube_index:estcube_index+3]
    estcube = Satellite(estcube_data, time_zone_offset=3)
    print(estcube.jul_to_date_time())
    # Prints everything needed for the first task
    estcube.print_params()
    """
    Propagate the satellite to any choosen date
    Does not take drag into account
    date should be in the form <yr month day> eg. 2024 April 6
    time should be in the form <hour:minute:second> without leading 0's eg. 16:0:0
    Steps is essentially the resolution in the time dimension.
    Code is not optimized by any stretch of the imagination, but 1000 steps runs insignificantly quickly
    """
    propagated_params = estcube.propagate_step(date="2024 April 12", time="12:0:0", steps=1000, supress=True)
    """
    Returns a dict with the keys: true_anomaly, mean_anomaly, eccentric_anomaly, geocentric_r_a,
    geocentric_declination, geocentric_cartesian, perigee_distance
    Some lists, some just one value
    """

    # Prints propagated params
    date_time = propagated_params.get("date")
    mean_anomaly = propagated_params.get("mean_anomaly")[-1]
    eccentric_anomaly = propagated_params.get("eccentric_anomaly")[-1]
    true_anomaly = propagated_params.get("true_anomaly")[-1]
    a = propagated_params.get("semi-major-axis")
    perigee_dist = propagated_params.get("perigee_distance")[-1]
    orbital_period = propagated_params.get("orbital_period")

    print(f"Propagated params:")
    print(f"Date, time: {date_time}")
    print(f"Mean anomaly: {mean_anomaly} rads")
    print(f"Eccentric anomaly {eccentric_anomaly} rads")
    print(f"True anomaly: {true_anomaly} rads")
    print(f"Semi major axis: {a} km")
    print(f"Perigee distance: {perigee_dist} km")
    print(f"Orbital period: {orbital_period} Minutes")
    
    # trajectory can be plotted:
    plotter = Plotter(propagated_params["geocentric_cartesian"])
    plotter.plot()

    # Satellite can also be propped to a certain timestamp without any intermediate steps
    # propped = estcube.propagate(date="2024 April 12", time="16:0:0")

if __name__ == "__main__":
    main()
