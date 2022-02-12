# This file was automatically generated by SWIG (http://www.swig.org).
# Version 4.0.2
#
# Do not make changes to this file unless you know what you are doing--modify
# the SWIG interface file instead.

from sys import version_info as _swig_python_version_info
if _swig_python_version_info < (2, 7, 0):
    raise RuntimeError("Python 2.7 or later required")

# Import the low-level C/C++ module
if __package__ or "." in __name__:
    from . import _spa
else:
    import _spa

try:
    import builtins as __builtin__
except ImportError:
    import __builtin__

def _swig_repr(self):
    try:
        strthis = "proxy of " + self.this.__repr__()
    except __builtin__.Exception:
        strthis = ""
    return "<%s.%s; %s >" % (self.__class__.__module__, self.__class__.__name__, strthis,)


def _swig_setattr_nondynamic_instance_variable(set):
    def set_instance_attr(self, name, value):
        if name == "thisown":
            self.this.own(value)
        elif name == "this":
            set(self, name, value)
        elif hasattr(self, name) and isinstance(getattr(type(self), name), property):
            set(self, name, value)
        else:
            raise AttributeError("You cannot add instance attributes to %s" % self)
    return set_instance_attr


def _swig_setattr_nondynamic_class_variable(set):
    def set_class_attr(cls, name, value):
        if hasattr(cls, name) and not isinstance(getattr(cls, name), property):
            set(cls, name, value)
        else:
            raise AttributeError("You cannot add class attributes to %s" % cls)
    return set_class_attr


def _swig_add_metaclass(metaclass):
    """Class decorator for adding a metaclass to a SWIG wrapped class - a slimmed down version of six.add_metaclass"""
    def wrapper(cls):
        return metaclass(cls.__name__, cls.__bases__, cls.__dict__.copy())
    return wrapper


class _SwigNonDynamicMeta(type):
    """Meta class to enforce nondynamic attributes (no new attributes) for a class"""
    __setattr__ = _swig_setattr_nondynamic_class_variable(type.__setattr__)


SPA_ZA = _spa.SPA_ZA
SPA_ZA_INC = _spa.SPA_ZA_INC
SPA_ZA_RTS = _spa.SPA_ZA_RTS
SPA_ALL = _spa.SPA_ALL
class spa_data(object):
    thisown = property(lambda x: x.this.own(), lambda x, v: x.this.own(v), doc="The membership flag")
    __repr__ = _swig_repr
    year = property(_spa.spa_data_year_get, _spa.spa_data_year_set)
    month = property(_spa.spa_data_month_get, _spa.spa_data_month_set)
    day = property(_spa.spa_data_day_get, _spa.spa_data_day_set)
    hour = property(_spa.spa_data_hour_get, _spa.spa_data_hour_set)
    minute = property(_spa.spa_data_minute_get, _spa.spa_data_minute_set)
    second = property(_spa.spa_data_second_get, _spa.spa_data_second_set)
    delta_ut1 = property(_spa.spa_data_delta_ut1_get, _spa.spa_data_delta_ut1_set)
    delta_t = property(_spa.spa_data_delta_t_get, _spa.spa_data_delta_t_set)
    time_zone = property(_spa.spa_data_time_zone_get, _spa.spa_data_time_zone_set)
    longitude = property(_spa.spa_data_longitude_get, _spa.spa_data_longitude_set)
    latitude = property(_spa.spa_data_latitude_get, _spa.spa_data_latitude_set)
    elevation = property(_spa.spa_data_elevation_get, _spa.spa_data_elevation_set)
    pressure = property(_spa.spa_data_pressure_get, _spa.spa_data_pressure_set)
    temperature = property(_spa.spa_data_temperature_get, _spa.spa_data_temperature_set)
    slope = property(_spa.spa_data_slope_get, _spa.spa_data_slope_set)
    azm_rotation = property(_spa.spa_data_azm_rotation_get, _spa.spa_data_azm_rotation_set)
    atmos_refract = property(_spa.spa_data_atmos_refract_get, _spa.spa_data_atmos_refract_set)
    function = property(_spa.spa_data_function_get, _spa.spa_data_function_set)
    jd = property(_spa.spa_data_jd_get, _spa.spa_data_jd_set)
    jc = property(_spa.spa_data_jc_get, _spa.spa_data_jc_set)
    jde = property(_spa.spa_data_jde_get, _spa.spa_data_jde_set)
    jce = property(_spa.spa_data_jce_get, _spa.spa_data_jce_set)
    jme = property(_spa.spa_data_jme_get, _spa.spa_data_jme_set)
    l = property(_spa.spa_data_l_get, _spa.spa_data_l_set)
    b = property(_spa.spa_data_b_get, _spa.spa_data_b_set)
    r = property(_spa.spa_data_r_get, _spa.spa_data_r_set)
    theta = property(_spa.spa_data_theta_get, _spa.spa_data_theta_set)
    beta = property(_spa.spa_data_beta_get, _spa.spa_data_beta_set)
    x0 = property(_spa.spa_data_x0_get, _spa.spa_data_x0_set)
    x1 = property(_spa.spa_data_x1_get, _spa.spa_data_x1_set)
    x2 = property(_spa.spa_data_x2_get, _spa.spa_data_x2_set)
    x3 = property(_spa.spa_data_x3_get, _spa.spa_data_x3_set)
    x4 = property(_spa.spa_data_x4_get, _spa.spa_data_x4_set)
    del_psi = property(_spa.spa_data_del_psi_get, _spa.spa_data_del_psi_set)
    del_epsilon = property(_spa.spa_data_del_epsilon_get, _spa.spa_data_del_epsilon_set)
    epsilon0 = property(_spa.spa_data_epsilon0_get, _spa.spa_data_epsilon0_set)
    epsilon = property(_spa.spa_data_epsilon_get, _spa.spa_data_epsilon_set)
    del_tau = property(_spa.spa_data_del_tau_get, _spa.spa_data_del_tau_set)
    lamda = property(_spa.spa_data_lamda_get, _spa.spa_data_lamda_set)
    nu0 = property(_spa.spa_data_nu0_get, _spa.spa_data_nu0_set)
    nu = property(_spa.spa_data_nu_get, _spa.spa_data_nu_set)
    alpha = property(_spa.spa_data_alpha_get, _spa.spa_data_alpha_set)
    delta = property(_spa.spa_data_delta_get, _spa.spa_data_delta_set)
    h = property(_spa.spa_data_h_get, _spa.spa_data_h_set)
    xi = property(_spa.spa_data_xi_get, _spa.spa_data_xi_set)
    del_alpha = property(_spa.spa_data_del_alpha_get, _spa.spa_data_del_alpha_set)
    delta_prime = property(_spa.spa_data_delta_prime_get, _spa.spa_data_delta_prime_set)
    alpha_prime = property(_spa.spa_data_alpha_prime_get, _spa.spa_data_alpha_prime_set)
    h_prime = property(_spa.spa_data_h_prime_get, _spa.spa_data_h_prime_set)
    e0 = property(_spa.spa_data_e0_get, _spa.spa_data_e0_set)
    del_e = property(_spa.spa_data_del_e_get, _spa.spa_data_del_e_set)
    e = property(_spa.spa_data_e_get, _spa.spa_data_e_set)
    eot = property(_spa.spa_data_eot_get, _spa.spa_data_eot_set)
    srha = property(_spa.spa_data_srha_get, _spa.spa_data_srha_set)
    ssha = property(_spa.spa_data_ssha_get, _spa.spa_data_ssha_set)
    sta = property(_spa.spa_data_sta_get, _spa.spa_data_sta_set)
    zenith = property(_spa.spa_data_zenith_get, _spa.spa_data_zenith_set)
    azimuth_astro = property(_spa.spa_data_azimuth_astro_get, _spa.spa_data_azimuth_astro_set)
    azimuth = property(_spa.spa_data_azimuth_get, _spa.spa_data_azimuth_set)
    incidence = property(_spa.spa_data_incidence_get, _spa.spa_data_incidence_set)
    suntransit = property(_spa.spa_data_suntransit_get, _spa.spa_data_suntransit_set)
    sunrise = property(_spa.spa_data_sunrise_get, _spa.spa_data_sunrise_set)
    sunset = property(_spa.spa_data_sunset_get, _spa.spa_data_sunset_set)

    def __init__(self):
        _spa.spa_data_swiginit(self, _spa.new_spa_data())
    __swig_destroy__ = _spa.delete_spa_data

# Register spa_data in _spa:
_spa.spa_data_swigregister(spa_data)


def deg2rad(degrees):
    return _spa.deg2rad(degrees)

def rad2deg(radians):
    return _spa.rad2deg(radians)

def limit_degrees(degrees):
    return _spa.limit_degrees(degrees)

def third_order_polynomial(a, b, c, d, x):
    return _spa.third_order_polynomial(a, b, c, d, x)

def geocentric_right_ascension(lamda, epsilon, beta):
    return _spa.geocentric_right_ascension(lamda, epsilon, beta)

def geocentric_declination(beta, epsilon, lamda):
    return _spa.geocentric_declination(beta, epsilon, lamda)

def observer_hour_angle(nu, longitude, alpha_deg):
    return _spa.observer_hour_angle(nu, longitude, alpha_deg)

def right_ascension_parallax_and_topocentric_dec(latitude, elevation, xi, h, delta, delta_alpha, delta_prime):
    return _spa.right_ascension_parallax_and_topocentric_dec(latitude, elevation, xi, h, delta, delta_alpha, delta_prime)

def topocentric_right_ascension(alpha_deg, delta_alpha):
    return _spa.topocentric_right_ascension(alpha_deg, delta_alpha)

def topocentric_local_hour_angle(h, delta_alpha):
    return _spa.topocentric_local_hour_angle(h, delta_alpha)

def topocentric_elevation_angle(latitude, delta_prime, h_prime):
    return _spa.topocentric_elevation_angle(latitude, delta_prime, h_prime)

def atmospheric_refraction_correction(pressure, temperature, atmos_refract, e0):
    return _spa.atmospheric_refraction_correction(pressure, temperature, atmos_refract, e0)

def topocentric_elevation_angle_corrected(e0, delta_e):
    return _spa.topocentric_elevation_angle_corrected(e0, delta_e)

def topocentric_zenith_angle(e):
    return _spa.topocentric_zenith_angle(e)

def topocentric_azimuth_angle_astro(h_prime, latitude, delta_prime):
    return _spa.topocentric_azimuth_angle_astro(h_prime, latitude, delta_prime)

def topocentric_azimuth_angle(azimuth_astro):
    return _spa.topocentric_azimuth_angle(azimuth_astro)

def spa_calculate(spa):
    return _spa.spa_calculate(spa)


