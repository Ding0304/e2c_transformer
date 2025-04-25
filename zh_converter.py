from copy import deepcopy
import re

try:
    from zh_wiki import zh2Hant, zh2Hans  # 导入繁体和简体映射表
except ImportError:
    from zhtools.zh_wiki import zh2Hant, zh2Hans

import sys
py3k = sys.version_info >= (3, 0, 0)  # 检查是否为 Python 3.x

if py3k:
    UEMPTY = ''  # Python 3 中的空字符串
else:
    _zh2Hant, _zh2Hans = {}, {}
    for old, new in ((zh2Hant, _zh2Hant), (zh2Hans, _zh2Hans)):
        for k, v in old.items():
            new[k.decode('utf8')] = v.decode('utf8')
    zh2Hant = _zh2Hant
    zh2Hans = _zh2Hans
    UEMPTY = ''.decode('utf8')  # Python 2 中的空字符串

# 定义状态
(START, END, FAIL, WAIT_TAIL) = list(range(4))
# 定义条件
(TAIL, ERROR, MATCHED_SWITCH, UNMATCHED_SWITCH, CONNECTOR) = list(range(5))

MAPS = {}  # 全局映射表


class Node(object):
    """节点类，表示转换图中的一个节点"""
    def __init__(self, from_word, to_word=None, is_tail=True, have_child=False):
        self.from_word = from_word  # 原始词
        if to_word is None:
            self.to_word = from_word  # 如果没有目标词，则默认与原始词相同
            self.data = (is_tail, have_child, from_word)
            self.is_original = True  # 是否为原始词
        else:
            self.to_word = to_word or from_word
            self.data = (is_tail, have_child, to_word)
            self.is_original = False
        self.is_tail = is_tail  # 是否是尾节点
        self.have_child = have_child  # 是否有子节点

    def is_original_long_word(self):
        """判断是否为原始长词"""
        return self.is_original and len(self.from_word) > 1

    def is_follow(self, chars):
        """判断是否为跟随字符"""
        return chars != self.from_word[:-1]

    def __str__(self):
        return '<Node, %s, %s, %s, %s>' % (
            repr(self.from_word), repr(self.to_word), self.is_tail, self.have_child)

    __repr__ = __str__


class ConvertMap(object):
    """转换映射类，用于管理字符映射表"""
    def __init__(self, name, mapping=None):
        self.name = name  # 映射名称
        self._map = {}  # 映射表
        if mapping:
            self.set_convert_map(mapping)

    def set_convert_map(self, mapping):
        """设置转换映射表"""
        convert_map = {}
        have_child = {}
        max_key_length = 0
        for key in sorted(mapping.keys()):  # 遍历映射表
            if len(key) > 1:  # 处理多字符键
                for i in range(1, len(key)):
                    parent_key = key[:i]
                    have_child[parent_key] = True
            have_child[key] = False
            max_key_length = max(max_key_length, len(key))
        for key in sorted(have_child.keys()):
            convert_map[key] = (key in mapping, have_child[key], mapping.get(key, UEMPTY))
        self._map = convert_map
        self.max_key_length = max_key_length  # 记录最大键长度

    def __getitem__(self, k):
        """通过键获取对应的节点"""
        try:
            is_tail, have_child, to_word = self._map[k]
            return Node(k, to_word, is_tail, have_child)
        except:
            return Node(k)

    def __contains__(self, k):
        """判断键是否在映射表中"""
        return k in self._map

    def __len__(self):
        """返回映射表的长度"""
        return len(self._map)


class StatesMachineException(Exception):
    """状态机异常类"""
    pass


class StatesMachine(object):
    """状态机类，用于逐字符处理输入文本"""
    def __init__(self):
        self.state = START  # 当前状态
        self.final = UEMPTY  # 最终结果
        self.len = 0  # 当前处理的长度
        self.pool = UEMPTY  # 缓存池

    def clone(self, pool):
        """克隆当前状态机"""
        new = deepcopy(self)
        new.state = WAIT_TAIL
        new.pool = pool
        return new

    def feed(self, char, map):
        """处理输入字符"""
        node = map[self.pool + char]  # 获取当前字符对应的节点
        # 根据节点属性判断条件
        if node.have_child:
            if node.is_tail:
                cond = UNMATCHED_SWITCH if node.is_original else MATCHED_SWITCH
            else:
                cond = CONNECTOR
        else:
            cond = TAIL if node.is_tail else ERROR

        new = None
        if cond == ERROR:
            self.state = FAIL
        elif cond == TAIL:
            if self.state == WAIT_TAIL and node.is_original_long_word():
                self.state = FAIL
            else:
                self.final += node.to_word
                self.len += 1
                self.pool = UEMPTY
                self.state = END
        elif self.state == START or self.state == WAIT_TAIL:
            if cond == MATCHED_SWITCH:
                new = self.clone(node.from_word)
                self.final += node.to_word
                self.len += 1
                self.state = END
                self.pool = UEMPTY
            elif cond == UNMATCHED_SWITCH or cond == CONNECTOR:
                if self.state == START:
                    new = self.clone(node.from_word)
                    self.final += node.to_word
                    self.len += 1
                    self.state = END
                else:
                    if node.is_follow(self.pool):
                        self.state = FAIL
                    else:
                        self.pool = node.from_word
        elif self.state == END:
            self.state = START
            new = self.feed(char, map)
        elif self.state == FAIL:
            raise StatesMachineException('状态机处理输入数据时发生错误：%s' % node)
        return new

    def __len__(self):
        """返回状态机的长度"""
        return self.len + 1

    def __str__(self):
        """返回状态机的字符串表示"""
        return '<StatesMachine %s, 缓存池: "%s", 状态: %s, 最终结果: %s>' % (
            id(self), self.pool, self.state, self.final)

    __repr__ = __str__


class Converter(object):
    """转换器类，负责整体转换流程"""
    def __init__(self, to_encoding):
        self.to_encoding = to_encoding  # 目标编码
        self.map = MAPS[to_encoding]  # 获取对应映射表
        self.start()

    def feed(self, char):
        """逐字符处理输入"""
        branches = []
        for fsm in self.machines:
            new = fsm.feed(char, self.map)
            if new:
                branches.append(new)
        if branches:
            self.machines.extend(branches)
        self.machines = [fsm for fsm in self.machines if fsm.state != FAIL]
        all_ok = True
        for fsm in self.machines:
            if fsm.state != END:
                all_ok = False
        if all_ok:
            self._clean()
        return self.get_result()

    def _clean(self):
        """清理并生成最终结果"""
        if len(self.machines):
            self.machines.sort(key=lambda x: len(x))
            self.final += self.machines[0].final
        self.machines = [StatesMachine()]

    def start(self):
        """初始化状态机"""
        self.machines = [StatesMachine()]
        self.final = UEMPTY

    def end(self):
        """结束处理"""
        self.machines = [fsm for fsm in self.machines
                         if fsm.state == FAIL or fsm.state == END]
        self._clean()

    def convert(self, string):
        """执行转换"""
        self.start()
        for char in string:
            self.feed(char)
        self.end()
        return self.get_result()

    def get_result(self):
        """获取最终结果"""
        return self.final


def registery(name, mapping):
    """注册映射表"""
    global MAPS
    MAPS[name] = ConvertMap(name, mapping)

registery('zh-hant', zh2Hant)  # 注册繁体映射表
registery('zh-hans', zh2Hans)  # 注册简体映射表
del zh2Hant, zh2Hans  # 删除临时变量


def run():
    """主程序入口"""
    import sys
    from optparse import OptionParser
    parser = OptionParser()
    parser.add_option('-e', type='string', dest='encoding',
                      help='目标编码（如 zh-hant 或 zh-hans）')
    parser.add_option('-f', type='string', dest='file_in',
                      help='输入文件（- 表示标准输入）')
    parser.add_option('-t', type='string', dest='file_out',
                      help='输出文件（- 表示标准输出）')
    (options, args) = parser.parse_args()
    if not options.encoding:
        parser.error('必须指定目标编码')
    if options.file_in:
        if options.file_in == '-':
            file_in = sys.stdin
        else:
            file_in = open(options.file_in)
    else:
        file_in = sys.stdin
    if options.file_out:
        if options.file_out == '-':
            file_out = sys.stdout
        else:
            file_out = open(options.file_out, 'wb')
    else:
        file_out = sys.stdout

    c = Converter(options.encoding)
    for line in file_in:
        file_out.write(c.convert(line.rstrip('\n').decode(
            'utf8')).encode('utf8'))


if __name__ == '__main__':
    run()