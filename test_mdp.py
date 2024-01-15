from unittest import TestCase
import mdp

class TestMDP(TestCase):
    def test_init(self):
        m = mdp.MDP(mapfile='rnGraph')
        print(m)
    def test_computeEU(self):
        m = mdp.MDP(mapfile='rnGraph')
        print(m.computeEU("3"))

    def test_computePolicy(self):
        m = mdp.MDP(mapfile='rnGraph')
        m.utilities["1"] = 0.23
        m.utilities["2"] = 0.45
        m.utilities["3"] = 0.68
        m.utilities["4"] = 0.06
        m.utilities["5"] = 0.33
        m.utilities["6"] = -0.03
        m.utilities["7"] = -0.01
        m.utilities["8"] = 0.13
        m.utilities["9"] = -0.07
        m.computePolicy()
        ## should generate the same policy as in slide 73
