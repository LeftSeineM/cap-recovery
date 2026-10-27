"""Explicit acceptance scope from scene 000103.

Keep the excluded instances and their reasons in the test profile: a scene
contains background fixtures as well as movable objects, and full formal
runs must have a reproducible expected asset set. This file changes no policy.
"""

SCENE = "000103"
ROBOT = "x1pro_rm001"
TRIALS = 20
EXCLUDED_INSTANCES = {
    'backboard': '固定工作台背景',
    'monitor': '工作台固定设备，本轮不搬运',
    'psu': '工作台固定设备，本轮不搬运',
    'scope': '工作台固定设备，本轮不搬运',
    'box': '工作台固定设备，本轮不搬运',
    'lamp': '工作台固定设备，本轮不搬运',
    'station': '工作台固定设备，本轮不搬运',
    'holder': '工作台固定设备，本轮不搬运',
    'chest': '柜体与抽屉操作不属于整物抓起测试',
    'bin': '落地容器不属于本轮桌面小物抓起测试',
}
