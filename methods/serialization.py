import json

class BBox(object):
    def __init__(self, idx, name, prob, minx, maxx, miny, maxy):
        self.idx = idx  # index
        self.name = name    # 类别名称
        self.prob = prob    # 置信度
        self.minx = minx    # 检测框x方向最小值
        self.maxx = maxx    # 检测框x方向最小值
        self.miny = miny    # 检测框y方向最小值
        self.maxy = maxy    # 检测框y方向最大值

    def to_dict(self):
        jsondict = \
        {   'idx': self.idx,
            'name': self.name, 
            'prob': self.prob,
            'minx': self.minx,
            'maxx': self.maxx,
            'miny': self.miny,
            'maxy': self.maxy
        }
        return jsondict

    def to_json(self):
        jsondict = self.to_dict()
        return json.dumps(jsondict)


class DetectionResult(object):
    def __init__(self, img_tag, img_time, img_height, img_width, pre_time, inf_time, objs):
        self.img_tag = img_tag  # 图像标签
        self.img_time = img_time    # 成像时间
        self.img_height = img_height    # 图像高度
        self.img_width = img_width  # 图像宽度
        self.pre_time = pre_time    # 预处理时间
        self.inf_time = inf_time    # 推理时间
        self.list = objs    # 检测结果的列表，为BBox的list

    def to_dict(self):
        items = []
        for item in self.list:
            items.append(item.to_dict())
        jsondict = \
        {
            'img_tag': self.img_tag,
            'img_time': self.img_time,
            'img_height': self.img_height,
            'img_width': self.img_width,
            'pre_time': self.pre_time,
            'inf_time': self.inf_time,
            'list': items
        }
        return jsondict

    def to_json(self):
        jsondict = self.to_dict()
        return json.dumps(jsondict)
