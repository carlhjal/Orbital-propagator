import numpy as np

class Satellite:
    """
    Unpacks TLE data and puts it in a satellite object
    """
    def __init__(self, tle, leap_year=True, time_zone_offset=3):
        # Unpacking tle data
        self.leap_year = leap_year
        self.time_zone_offset = time_zone_offset
        self.name = tle[0]
        self.epoch_year = int(tle[1][18:20].strip())
        self.epoch_day = float(tle[1][20:32].strip().lstrip("0"))
        self.first_time_der = tle[1][33:43].strip() # drag
        self.second_time_der = tle[1][44:52].strip()
        self.bstar_drag = tle[1][53:61].strip()
        self.inclination = float(tle[2][8:16].strip()) #degrees
        self.right_ascension = float(tle[2][17:25].strip()) # of the ascending node
        self.eccentricity = float("0." + tle[2][26:33].strip())
        self.perigee_arg = float(tle[2][34:42].strip())
        self.mean_anomaly = float(tle[2][43:51].strip())
        self.mean_motion = float(tle[2][52:63].strip())
        self.rev_number = tle[2][63:68].strip()

        self.geocentric_distance = -1
        self.geocentric_r_a = -1
        self.geocentric_declination = -1
        self.a = -1
        
        # Simple conversion of mean motion to orbital time in minutes
        self.orbital_period_minutes = 24*60/(self.mean_motion)

        # rad conversions:
        self.right_ascension_rads = self.right_ascension*np.pi/180
        self.perigee_arg_rads = self.perigee_arg*np.pi/180
        self.mean_anomaly_rads = self.mean_anomaly*np.pi/180
        self.inclination_rads = self.inclination*np.pi/180

        # converting time to unix time:
        # epoch_day counts from 1
        s_in_year = round((self.epoch_day-1)*24*60*60)
        yr_diff = self.epoch_year + 2000 - 1970
        # -2 years since first leap year since unix time start was 1972
        # day epoch is in UTC time so + 3 hours for estonian summer time
        self.unix_time = s_in_year + (yr_diff*365*24*60*60) + ((yr_diff-2)//4*24*60*60) + (self.time_zone_offset*60*60)

        self.months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        self.days_in_months = [31,28+int(self.leap_year),31,30,31,30,31,31,30,31,30,31]

    def jul_to_date_time(self):
        """
        Returns a date and time for a TLE jul date
        """
        curr_month = ""
        curr_day = 0
        day_counter = int(self.epoch_day)

        # timestamp calculation
        day_frac = self.epoch_day - day_counter
        hour = day_frac*24 + self.time_zone_offset
        rem = hour - int(hour)
        minute = rem*60
        rem = minute - int(minute)
        # unlike minute and hour, second should be rounded up
        sec = int(round(rem*60))
        hour = int(hour)
        minute = int(minute)
        time = str(hour) +":"+str(minute)+":"+str(sec)

        # date calculation
        for month, days in enumerate(self.days_in_months):
            if days < day_counter:
                day_counter -= days
                continue
            curr_day = day_counter
            curr_month = self.months[month]
            break

        date = str(2000+self.epoch_year)+" "+curr_month+" "+str(curr_day)
        return date, time
    
    def unix_to_date_time(self, unix_time):
        """
        Returns a date and time for a particular unix time
        Does the same thing as the DateTime library
        """
        s_in_year = 365*24*60*60 + int(self.leap_year)*24*60*60
        year = 1970

        while True:
            s_in_year = 365*24*60*60 + int(not bool(year%4))*24*60*60
            if s_in_year > unix_time:
                break
            unix_time -= s_in_year
            year += 1
        
        s_in_months = [days*24*60*60 for days in self.days_in_months]
        
        month_num = 0
        for s_in_month in s_in_months:
            if s_in_month > unix_time:
                break
            unix_time -= s_in_month
            month_num += 1
        month = self.months[month_num]

        s_in_day = 24*60*60
        day = unix_time // s_in_day + 1
        unix_time -= (day-1)*s_in_day

        s_in_hour = 60*60
        hour = unix_time // s_in_hour
        unix_time -= hour*s_in_hour

        minute = unix_time // 60
        unix_time -= minute*60

        second = unix_time

        date = str(year)+' '+str(month)+' '+str(day)
        time = str(hour)+':'+str(minute)+':'+str(second)
        return date, time

    def date_time_to_unix(self, date, time):
        """
        Returns a date and time in unix time format without considering time zone
        Aka, the time zone will be whatever time zone of the date and time parameters passed
        """
        date_split = date.split(' ')
        time_split = time.split(':')
        assert len(date_split) == 3, "Something wrong with date"
        assert len(time_split) == 3, "Something wrong with time"

        yr = int(date_split[0])
        month = date_split[1]
        day = int(date_split[2])
        hour = int(time_split[0])
        minute = int(time_split[1])
        second = int(time_split[2])

        jul_day = 0.0

        for curr_month, days in enumerate(self.days_in_months):
            if self.months[curr_month].lower() == month.lower():
                jul_day += day
                break
            jul_day += days

        s_in_year = ((jul_day-1)*24*60*60) + (hour*60*60) + (minute*60) + second
        yr_diff = yr-1970
        unix_time = s_in_year + (yr_diff*365*24*60*60) + ((yr_diff-2)//4*24*60*60)

        return int(unix_time)

    def print_params(self, full=False): # for manual checking that everything parsed correctly
        if full is not True:
             date, time = self.jul_to_date_time()
             print(f"Date: {date}, time: {time}")
             print(f"Inclincation: {self.inclination_rads}")
             print(f"Right ascension of the ascending node: {self.right_ascension_rads} rads")
             print(f"Eccentricity: {self.eccentricity}")
             print(f"Argument of periapsis: {self.perigee_arg_rads} rads")
             print(f"Mean anomaly: {self.mean_anomaly_rads} rads")
             print(f"Mean motion: {self.mean_motion}")
        else:
            print(self.name)
            print(self.epoch_year)
            print(self.epoch_day)
            print(self.first_time_der)
            print(self.second_time_der)
            print(self.bstar_drag)
            print(self.inclination)
            print(self.right_ascension)
            print(self.eccentricity)
            print(self.perigee_arg)
            print(self.mean_anomaly)
            print(self.mean_motion)
            print(self.rev_number)
            print(self.unix_time)

    def propagate_step(self,date="2024 april 12", time="16:0:0", steps=100, supress=True):
        """
        Creates a linspace with <steps> number of steps between start time and end time
        Calls propagate iteratively for each time step

        Returns a list of cartesian coordinates 
        """
        unix_t = self.date_time_to_unix(date, time)
        time_steps = np.linspace(self.unix_time, unix_t, steps, dtype="uint32")
        time_steps_date_time = [self.unix_to_date_time(time_step) for time_step in time_steps]

        propagated_params = {"true_anomaly" : [],
                             "mean_anomaly" : [],
                             "eccentric_anomaly" : [],
                             "geocentric_r_a" : [],
                             "geocentric_declination" :[],
                             "geocentric_cartesian" : [],
                             "perigee_distance" : [],
                             }
        
        for date_time in time_steps_date_time:
            res = self.propagate(date=date_time[0], time=date_time[1], suppress=supress)
            propagated_params["true_anomaly"].append(res["true_anomaly"])
            propagated_params["mean_anomaly"].append(res["mean_anomaly"])
            propagated_params["eccentric_anomaly"].append(res["eccentric_anomaly"])
            propagated_params["geocentric_r_a"].append(res["geocentric_r_a"])
            propagated_params["geocentric_declination"].append(res["geocentric_declination"])
            propagated_params["geocentric_cartesian"].append(res["geocentric_cartesian"])
            propagated_params["perigee_distance"].append(res["perigee_distance"])
        
        propagated_params["date"] = date, time
        propagated_params["orbital_period"] = self.orbital_period_minutes
        propagated_params["semi-major-axis"] = self.a

        return propagated_params

    def propagate(self, date="2024 april 12", time="16:0:0", suppress=True):
        
        #params for testing and verification
        """
        self.eccentricity = 0.0001492
        self.mean_anomaly_rads = 326.2322*np.pi/180
        self.mean_motion = 12.62256095
        self.inclination = 51.9970
        self.inclination_rads = self.inclination*np.pi/180
        self.right_ascension_rads = 251.0219*np.pi/180
        self.perigee_arg_rads = 33.8641*np.pi/180
        dt = 1.7677141
        """

        dt = self.date_time_to_unix(date, time) - self.unix_time
        dt = dt/(24*60*60)

        M_t = self._calc_mean_anomaly_t(self.mean_anomaly_rads, self.mean_motion, dt)
        E = self.eccentric_anomaly_iteratively(self.eccentricity, M_t)
        true_anomaly = self._eccentric_to_true_anomaly(self.eccentricity, E)

        a = self._calc_semi_major_axis(self.mean_motion)
        self.a = a
        P = self._calc_perigee_distance(a,self.eccentricity)
        r_geo = self._calc_geocentric_distance(P, self.eccentricity, true_anomaly)

        # average Earth radius (equatorial+polar/2)
        # since satellite is in a somewhat polar orbit
        r_e = 1 #(6378 + 6357) / 2, in units of Earth radii
        j_2 = 1.0826267e-3 # second gravitational zonal harmonic of the Earth

        p_0 = self._calc_p_0(a, r_e, j_2, self.inclination_rads, self.eccentricity)
        r_a_precessed = self._calc_ascending_node_precession(j_2, r_e, dt, self.mean_motion, self.inclination_rads, self.right_ascension_rads, p_0)
        perigee_precessed = self._calc_perigee_precession(j_2, r_e, dt, self.mean_motion, self.inclination_rads, self.perigee_arg_rads, p_0)

        arg_of_latitude = self._calc_arg_of_latitude(perigee_precessed, true_anomaly)
        r_a_difference = self._calc_r_a_difference(arg_of_latitude, self.inclination_rads)
        geocentric_r_a = self._calc_geocentric_right_ascension(r_a_difference, r_a_precessed)
        geocentric_declination = self._calc_geocentric_declination(arg_of_latitude, r_a_difference)

        self.geocentric_distance = r_geo
        self.geocentric_r_a = geocentric_r_a
        self.geocentric_declination = geocentric_declination

        cart = self.to_cart()

        if not suppress:
            print(f"dt in solar days: {dt}")
            print(f"semi major axis: {a}")
            print(f"perigee_distance: {P}")
            print(f"geocentric_distance: {r_geo}")
            print(f"precessed R.A of ascending node: {r_a_precessed}")
            print(f"precessed argument of perigee: {perigee_precessed}")
            print(f"argument of latitude: {arg_of_latitude}")
            print(f"R.A. difference: {r_a_difference}")
            print(f"Geocentric R.A. {geocentric_r_a}")
            print(f"Geocentric declination {geocentric_declination}")
            print(f"cartesian coordinates, x: {cart[0]}, y: {cart[1]}, z: {cart[2]}")

        # comment out this line
        #cart = [round(cart[0]), round(cart[1]), round(cart[2])]

        propped_params = {"true_anomaly" : true_anomaly,
                          "mean_anomaly" : M_t,
                          "eccentric_anomaly" : E,
                          "geocentric_r_a" : geocentric_r_a,
                          "geocentric_declination" : geocentric_declination,
                          "geocentric_cartesian" : cart,
                          "perigee_distance" : perigee_precessed}

        return propped_params

    def from_cart(self, coordinates):
        return np.sqrt(coordinates[0]**2 + coordinates[1]**2 + coordinates[2]**2)

    """
    Formulas for all calculations if not credited elsewhere are taken from
    http://www.castor2.ca/04_Propagation/index.html
    """

    def to_cart(self):
        """
        Covert geocentric R.A and declination to cartesian coordinates
        """
        x = self.geocentric_distance*np.cos(self.geocentric_r_a)*np.cos(self.geocentric_declination)
        y = self.geocentric_distance*np.sin(self.geocentric_r_a)*np.cos(self.geocentric_declination)
        z = self.geocentric_distance*np.sin(self.geocentric_declination)
        return x,y,z

    def _calc_geocentric_declination(self, arg_of_latitude, r_a_difference):
        geocentric_declination = np.sign(np.sin(arg_of_latitude))*np.arccos(np.cos(arg_of_latitude)/np.cos(r_a_difference))
        assert -np.pi/2 < geocentric_declination < np.pi/2
        return geocentric_declination

    def _calc_geocentric_right_ascension(self, r_a_difference, r_a_precessed):
        geocentric_r_a = r_a_difference + r_a_precessed - 2*np.pi*(int((r_a_difference+r_a_precessed)/(2*np.pi)))
        return geocentric_r_a

    def _calc_r_a_difference(self, arg_of_latitude, i):
        sub = np.cos(arg_of_latitude)/np.sqrt(1-np.sin(i)**2*np.sin(arg_of_latitude)**2)
        if (0 < i < np.pi/2) and (0 < arg_of_latitude < np.pi) or (np.pi/2 < i < np.pi) and (np.pi < arg_of_latitude < 2*np.pi):
            return np.arccos(sub)
        return (2*np.pi)-np.arccos(sub)

    def _calc_arg_of_latitude(self, perigee_precessed, true_anomaly):
        return perigee_precessed + true_anomaly - ((2*np.pi) * int((perigee_precessed + true_anomaly)/(2*np.pi)))

    def _calc_p_0(self, a, r_e, j_2, i, e):
        a_1 = a/((6378 + 6357) / 2)
        d_1 = (3*j_2*r_e**2*(3*np.cos(i)**2-1)) / ((4*a_1**2*(1-e**2))**(3/2))
        a_0 = -a_1*((134*d_1**3/81) + d_1**2 + d_1/3 -1)
        p_0 = a_0 * (1-e**2)
        return p_0

    def _calc_ascending_node_precession(self, j_2, r_e, dt, mean_motion, i, ra_of_ascending_node, p_0):
        d_a_omega = 2*np.pi*(-3*j_2*r_e**2*mean_motion*dt*np.cos(i)/(2*p_0**2))
        return ra_of_ascending_node + d_a_omega

    def _calc_perigee_precession(self, j_2, r_e, dt, mean_motion, i, perigee_arg, p_0):
        d_omega = 2*np.pi*(3*j_2*r_e**2*mean_motion*dt*(5*np.cos(i)**2 -1)/(4*p_0**2))
        return perigee_arg + d_omega

    def _calc_geocentric_distance(self, P, e, true_anom):
        return (P * (1+e)) / (1+e*np.cos(true_anom))
    
    def _calc_semi_major_axis(self, mean_motion):
        return ((2.97554e15) / ((2*np.pi*mean_motion)**2))**(1/3)
    
    def _calc_perigee_distance(self, a, e):
        return a*(1-e)

    def _calc_mean_anomaly_t(self, M_0, mean_motion_0, dt):
        sub = mean_motion_0*dt
        mean_anom_t = M_0 + (2*np.pi)*(sub - int(sub) - int((M_0+2*np.pi*(sub - int(sub)))/(2*np.pi)))
        assert mean_anom_t <= 2*np.pi
        return M_0 + ((2*np.pi)*(sub - int(sub) - int((M_0+2*np.pi*(sub - int(sub)))/(2*np.pi))))

    def _eccentric_to_true_anomaly(self, e, E):
        """
        Equation taken from https://www.johndcook.com/blog/2022/10/22/orbital-anomalies/
        """
        return E + 2*np.arctan(e*np.sin(E)/(1-e*np.cos(E)))
        
    def eccentric_anomaly(self, e, M):
        """
        Third-order guess of the eccentric anomaly given eccentricity and mean anomaly
        Marc A. Murison "A practical method for solving the Kepler equation"
        """
        E = (M + e*np.sin(M) + e**2*np.sin(M)*np.cos(M)+(0.5*e**3*np.sin(M))*(3*np.cos(M)*np.cos(M)-1))
        return E
    
    def _iteration_step(self, e, M, e_last):
        """
        Marc A. Murison "A practical method for solving the Kepler equation"
        """
        t1 = np.cos(e_last)
        t2 = -1 + e*t1
        t3 = np.sin(e_last)
        t4 = e*t3
        t5 = -e_last + t4 + M
        t6 = t5/(0.5 * t5 * t4/ t2 + t2)
        return t5/((0.5 * t3 - 1/6*t1*t6)*e*t6+t2)
    
    def eccentric_anomaly_iteratively(self, e, M, tolerance=1.0*10**-10, steps=100):
        """
        Marc A. Murison "A practical method for solving the Kepler equation"
        """
        E = 0
        M_norm = M%(2*np.pi)
        E_0 = self.eccentric_anomaly(e, M)
        d_E = tolerance + 1
        count = 0
        while d_E > tolerance:
            E = E_0 - self._iteration_step(e, M_norm, E_0)
            d_E = np.abs(E-E_0)
            E_0 = E
            count += 1
            if count == steps:
                print("Failed to converge")
                break
        return E