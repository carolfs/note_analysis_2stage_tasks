# Copyright 2017 Carolina Feher da Silva <carolfsu@gmail.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Creates Figure "Difference in stay probability for model-based agents." """

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit

x = np.linspace(0, 2, 100)
for alpha in (0.2, 0.4, 0.6, 0.8):
    y = []
    for ppb in x:
        p = ppb*0.5
        b = p
        y.append(
            expit(p - b - alpha*p + alpha) - expit(p - b + alpha*b - alpha) +\
            expit(p - b - alpha*p) - expit(p - b + alpha*b))
    plt.plot(x, y, label=r'$\alpha = {}$'.format(alpha))
plt.xlabel(r'Sum of the reward probabilities at the final states $(p + b)$')
plt.ylabel(r'Difference in stay probability $[(P_{rc} + P_{uc}) - (P_{rr} + P_{ur})]$')
plt.legend()
plt.axhline(y=0, xmax=2, ls='dotted', c='k')
plt.axvline(x=1, ls='dotted', c='k')
plt.savefig('ppb_sumprobs.eps', bbox_inches='tight')
plt.close()
