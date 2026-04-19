class Core():
    """
    A class used to represent a Core in a benchmarking system.
    Attributes
    ----------
    id : int
        The unique identifier for the core.
    benchmark : str
        The benchmark associated with the core.
    core_no : int
        The core index.
    patterns : int
        Number of patterns associated with the core.
    scan : int
        Scan chain length for the core
    Methods
    -------
    __init__(self, id, benchmark, core_no, patterns, scan)
        Initializes the Core with the given id, benchmark, core number, patterns, and scanchain length
    """
    
    def __init__(self, id, benchmark, core_no, patterns, scan):
        self.id = id
        self.benchmark = benchmark
        self.core_no = core_no
        self.patterns = patterns
        self.scan = scan
        
class IOPair():
    """
    IOPair class represents a pair of input (src) and output (sink).

    Attributes:
        src: The source core index.
        sink: The sink core index.

    Methods:
        __init__(self, src, sink): Initializes the IOPair with a source and a sink.
    """
    def __init__(self, src, sink):
        self.src = src
        self.sink = sink