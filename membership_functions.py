from fuzzy_set import FuzzySet
import numpy as np

# ------------------------------------------------------------------------
# Working QUANTIZATION 1 (1 trial to learn)
# ------------------------------------------------------------------------
berenji_quantization_inputs = [
    {  # imf_x
        "ZE": FuzzySet(x_zero=0.1, x_one=0, sink_beyond_1=True, y_symmetry=True),
        "VS": FuzzySet(x_zero=0.2, x_one=0, sink_beyond_1=True, y_symmetry=True),
        "PO": FuzzySet(x_zero=0.15, x_one=1.5, sink_beyond_1=False, y_symmetry=False),
        "NE": FuzzySet(x_zero=-0.15, x_one=-1.5, sink_beyond_1=False, y_symmetry=False)
    },

    {  # imf_xd
        "ZE": FuzzySet(x_zero=0.1, x_one=0, sink_beyond_1=True, y_symmetry=True),
        "VS": FuzzySet(x_zero=0.2, x_one=0, sink_beyond_1=True, y_symmetry=True),
        "PO": FuzzySet(x_zero=0.15, x_one=0.5, sink_beyond_1=False, y_symmetry=False),
        "NE": FuzzySet(x_zero=-0.15, x_one=-0.5, sink_beyond_1=False, y_symmetry=False)
    },
    {  # imf_theta
        "ZE": FuzzySet(x_zero=np.deg2rad(0.2), x_one=0.0, sink_beyond_1=True, y_symmetry=True),
        "VS": FuzzySet(x_zero=np.deg2rad(1.0), x_one=0.0, sink_beyond_1=True, y_symmetry=True),
        "PO": FuzzySet(x_zero=np.deg2rad(0.3), x_one=np.deg2rad(1.5), sink_beyond_1=False, y_symmetry=False),
        "NE": FuzzySet(x_zero=np.deg2rad(-0.3), x_one=np.deg2rad(-1.5), sink_beyond_1=False, y_symmetry=False)
    },
    {  # imf_thetad
        "ZE": FuzzySet(x_zero=np.deg2rad(0.2), x_one=0.0, sink_beyond_1=True, y_symmetry=True),
        "VS": FuzzySet(x_zero=np.deg2rad(1.0), x_one=0.0, sink_beyond_1=True, y_symmetry=True),
        "PO": FuzzySet(x_zero=np.deg2rad(0.3), x_one=np.deg2rad(1.5), sink_beyond_1=False, y_symmetry=False),
        "NE": FuzzySet(x_zero=np.deg2rad(-0.3), x_one=np.deg2rad(-1.5), sink_beyond_1=False, y_symmetry=False)
    }
]

berenji_quantization_outputs = {  # output membership functions
    "PL": FuzzySet(x_zero=2.5, x_one=10.0, sink_beyond_1=True, y_symmetry=False),
    "PM": FuzzySet(x_zero=1.2, x_one=3.0, sink_beyond_1=True, y_symmetry=False),
    "PS": FuzzySet(x_zero=0.2, x_one=2.0, sink_beyond_1=True, y_symmetry=False),
    "PVS": FuzzySet(x_zero=1.0, x_one=0.0, sink_beyond_1=True, y_symmetry=False),
    "NVS": FuzzySet(x_zero=-1.0, x_one=0.0, sink_beyond_1=True, y_symmetry=False),
    "NS": FuzzySet(x_zero=-0.2, x_one=-2.0, sink_beyond_1=True, y_symmetry=False),
    "NM": FuzzySet(x_zero=-1.2, x_one=-3.0, sink_beyond_1=True, y_symmetry=False),
    "NL": FuzzySet(x_zero=-2.5, x_one=-10.0, sink_beyond_1=True, y_symmetry=False),
    "0": FuzzySet(x_zero=0, x_one=0.001, sink_beyond_1=True, y_symmetry=False),
}
