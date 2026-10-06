import pytz
from datetime import datetime


def time_string_2_timestamp(time_string):
    # 设置北京时区
    beijing_tz = pytz.timezone('Asia/Shanghai')

    # 将时间字符串转换为 datetime 对象
    dt_object = datetime.strptime(time_string, '%Y-%m-%d %H:%M:%S')

    # 将 datetime 对象转换为北京时间
    dt_object = beijing_tz.localize(dt_object)

    # 使用 timestamp() 将 datetime 对象转换为时间戳
    return int(dt_object.timestamp())


def timestamp_2_time_string(timestamp):
    # 设置北京时区
    beijing_tz = pytz.timezone('Asia/Shanghai')

    # 将时间戳转换为 datetime 对象
    dt_object = datetime.utcfromtimestamp(timestamp).replace(tzinfo=pytz.utc)

    # 将 datetime 对象转换为北京时间
    dt_object = dt_object.astimezone(beijing_tz)

    return dt_object.strftime("%Y-%m-%d %H:%M:%S")
