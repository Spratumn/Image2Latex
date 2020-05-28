import re
import numpy as np

tokenstr = [
			r'\\dot ', r'\\ddot ', r'\\acute ', r'\\grave ', r'\\check ', r'\\breve ', r'\\tilde ',
			r'\\bar ', r'\\hat ', r'\\widehat ', r'\\vec ',
			r'\\exp ', r'\\ln ', r'\\lg ', r'\\log ', r'\\sin ', r'\\cos ', r'\\tan ',
			r'\\cot ', r'\\sec ', r'\\csc ', r'\\arcsin ', r'\\arccos ', r'\\arctan ',
			r'\\arcsec ', r'\\arccsc ', r'\\sinh ', r'\\cosh ', r'\\tanh ', r'\\coth ',
			r'\\dot', r'\\ddot', r'\\acute', r'\\grave', r'\\check', r'\\breve', r'\\tilde',
			r'\\bar', r'\\hat', r'\\widehat', r'\\vec',
			r'\\exp', r'exp', r'\\ln', r'\\lg', r'\\log', r'\\sin', r'\\cos', r'\\tan',
			r'\\cot', r'\\sec', r'\\csc', r'\\arcsin', r'\\arccos', r'\\arctan',
			r'\\arcsec', r'\\arccsc', r'\\sinh', r'\\cosh', r'\\tanh', r'\\coth',

			r'\\operatorname ', r'\\left\vert ', r'\\right\vert ', r'\\vert ',
			r'\\min ', r'\\max ', r'\\inf ', r'\\sup ', r'\\lim ', r'\\liminf ', r'\\limsup ', r'\\dim ',
			r'\\deg ', r'\\det ', r'\\ker ',
			r'\\Pr ', r'\\hom ', r'\\lVert ', r'\\rVert ', r'\\arg ',
			r'dt ', r'\\mathrm ', r'\\partial ', r'\\nabla ', r'dy ',  r'dx ', r'\\frac ', r'\\frac',
			r'\\prime ', r'\\backprime ', r"' ", r"'' ", r"''' ",
			r'\\infty ', r'\\aleph ', r'\\complement ', r'\\backepsilon ', r'\\eth ', r'\\Finv ',
			r'\\hbar ', r'\\Im ', r'\\imath ', r'\\jmath ', r'\\Bbbk ', r'\\ell ', r'\\mho ',
			r'\\wp ', r'\\Re ', r'\\circled ', r'\\S ', r'\\P ', r'\\AA ',
			r'\\equiv ', r'\\pmod ', r'\\bmod ', r'\\gcd ', r'\\mid ', r'\\nmid ', r'\\shortmid ', r'\\nshortmi ',
			r'\\surd ', r'\\sqrt ',
			r'\\pm ', r'\\mp ', r'\\dotplus ', r'\\div ', r'\\times ', r'\\divideontimes ',
			r'\+', r'\-', r'\*', r'/', r'\\backslash ', r'\\cdot ', r'\\ast ', r'\\star ', r'\\circ ', r'\\bullet ',
			r'\\boxplus ', r'\\boxminus ', r'\\boxtimes ', r'\\boxdot ', r'\\oplus ', r'\\ominus ', r'\\otimes ',
			r'\\oslash ', r'\\odot ', r'\\circleddash ', r'\\circledcirc ', r'\\circledast ', r'\\bigoplus ',
			r'\\bigotimes ', r'\\bigodot ', r'\\label ',

			r'\\operatorname', r'\\left\vert', r'\\right\vert', r'\\vert',
			r'\\min', r'\\max', r'\\inf', r'\\sup', r'\\lim', r'\\liminf', r'\\limsup', r'\\dim',
			r'\\deg', r'\\det', r'\\ker',
			r'\\Pr', r'\\hom', r'\\lVert', r'\\rVert', r'\\arg',
			r'dt', r'\\mathrm', r'\\partial', r'\\nabla', r'dy',  r'dx',
			r'\\prime', r'\\backprime', r"'", r"''", r"'''",
			r'\\infty', r'\\aleph', r'\\complement', r'\\backepsilon', r'\\eth', r'\\Finv',
			r'\\hbar', r'\\Im', r'\\imath', r'\\jmath', r'\\Bbbk', r'\\ell', r'\\mho',
			r'\\wp', r'\\Re', r'\\circled', r'\\S', r'\\P', r'\\AA',
			r'\\equiv', r'\\pmod', r'\\bmod', r'\\gcd', r'\\mid', r'\\nmid', r'\\shortmid', r'\\nshortmi',
			r'\\surd', r'\\sqrt',
			r'\\pm', r'\\mp', r'\\dotplus', r'\\div', r'\\times', r'\\divideontimes',
			r'\\backslash', r'\\cdot', r'\\ast', r'\\star', r'\\circ', r'\\bullet',
			r'\\boxplus', r'\\boxminus', r'\\boxtimes', r'\\boxdot', r'\\oplus', r'\\ominus', r'\\otimes',
			r'\\oslash', r'\\odot', r'\\circleddash', r'\\circledcirc', r'\\circledast', r'\\bigoplus',
			r'\\bigotimes', r'\\bigodot', r'\\label',

			r'\\{', r'\\}', r'\\O ', r'\\empty ', r'\\emptyset ', r'\\varnothing ', r'\\in ', r'\\notin ', r'\\not\\in ',
			r'\\ni ', r'\\not\\ni ', r'\\cap ', r'\\Cap ', r'\\sqcap ', r'\\bigcap ', r'\\cup ', r'\\Cup ', r'\\sqcup ',
			r'\\bigcup ', r'\\bigsqcup ', r'\\uplus ', r'\\biguplus ', r'\\setminus ', r'\\smallsetminus ', r'\\subset ',
			r'\\Subset ', r'\\sqsubset ', r'\\supset ', r'\\Supset ', r'\\sqsupset ', r'\\subseteq ', r'\\nsubseteq ',
			r'\\subsetneq ', r'\\varsubsetneq ', r'\\sqsubseteq ', r'\\supseteq ', r'\\nsupseteq ', r'\\supsetneq ',
			r'\\varsupsetneq ', r'\\sqsupseteq ', r'\\subseteqq ', r'\\nsubseteqq ', r'\\subsetneqq ', r'\\varsubsetneqq ',
			r'\\supseteqq ', r'\\nsupseteqq ', r'\\supsetneqq ', r'\\varsupsetneqq ',
			r'\\ne ', r'\\neq ', r'\\not\\equiv ', r'\\doteq ', r'\\doteqdot ',

			r'\\O', r'\\empty', r'\\emptyset', r'\\varnothing', r'\\in', r'\\notin', r'\\not\\in',
			r'\\ni', r'\\not\\ni', r'\\cap', r'\\Cap', r'\\sqcap', r'\\bigcap', r'\\cup', r'\\Cup', r'\\sqcup',
			r'\\bigcup', r'\\bigsqcup', r'\\uplus', r'\\biguplus', r'\\setminus', r'\\smallsetminus', r'\\subset',
			r'\\Subset', r'\\sqsubset', r'\\supset', r'\\Supset', r'\\sqsupset', r'\\subseteq', r'\\nsubseteq',
			r'\\subsetneq', r'\\varsubsetneq', r'\\sqsubseteq', r'\\supseteq', r'\\nsupseteq', r'\\supsetneq',
			r'\\varsupsetneq', r'\\sqsupseteq', r'\\subseteqq', r'\\nsubseteqq', r'\\subsetneqq', r'\\varsubsetneqq',
			r'\\supseteqq', r'\\nsupseteqq', r'\\supsetneqq', r'\\varsupsetneqq',
			r'\\ne', r'\\neq', r'\\not\\equiv', r'\\doteq', r'\\doteqdot',

			r'\\overset{\\underset{\\mathrm{def}}{}}', r':=',
			r'\\sim ', r'\\nsim ', r'\\backsim ', r'\\thicksim ',
			r'\\simeq ', r'\\backsimeq ', r'\\eqsim ', r'\\cong ', r'\\ncong ', r'\\approx ', r'\\thickapprox ',
			r'\\approxeq ', r'\\asymp ', r'\\propto ', r'\\varpropto ', r'<', r'\\nless ', r'\\ll ',
			r'\\not\\ll ', r'\\lll ', r'\\not\\lll ', r'\\lessdot ', r'>', r'\\ngtr ', r'\\gg ', r'\\not\\gg ', r'\\ggg ',
			r'\\not\\ggg ', r'\\gtrdot ', r'\\le ', r'\\leq ', r'\\lneq ', r'\\leqq ',
			r'\\nleq ', r'\\nleqq ', r'\\lneqq ', r'\\lvertneqq ', r'\\ge ', r'\\geq ', r'\\gneq ',
			r'\\geqq ', r'\\ngeq ', r'\\ngeqq ', r'\\gneqq ', r'\\gvertneqq ', r'\\lessgtr ', r'\\lesseqgtr ',
			r'\\lesseqqgtr ', r'\\gtrless ', r'\\gtreqless ', r'\\gtreqqless ', r'\\leqslant ', r'\\nleqslant ',
			r'\\eqslantless ', r'\\geqslant ', r'\\ngeqslant ', r'\\eqslantgtr ', r'\\lesssim ', r'\\lnsim ',
			r'\\lessapprox ', r'\\lnapprox ', r'\\gtrsim ', r'\\gnsim ', r'\\gtrapprox ', r'\\gnapprox ',
			r'\\prec ', r'\\nprec ', r'\\preceq ', r'\\npreceq ', r'\\precneqq ', r'\\succ ', r'\\nsucc ',
			r'\\succeq ', r'\\nsucceq ', r'\\succneqq ', r'\\preccurlyeq ', r'\\curlyeqprec ', r'\\succcurlyeq ',
			r'\\curlyeqsucc ', r'\\precsim ', r'\\precnsim ', r'\\precapprox ', r'\\precnapprox ', r'\\succsim ',
			r'\\succnsim ', r'\\succapprox ', r'\\succnapprox ',

			r'\\sim', r'\\nsim', r'\\backsim', r'\\thicksim',
			r'\\simeq', r'\\backsimeq', r'\\eqsim', r'\\cong', r'\\ncong', r'\\approx', r'\\thickapprox',
			r'\\approxeq', r'\\asymp', r'\\propto', r'\\varpropto', r'\\nless', r'\\ll',
			r'\\not\\ll', r'\\lll', r'\\not\\lll', r'\\lessdot', r'\\ngtr', r'\\gg', r'\\not\\gg', r'\\ggg',
			r'\\not\\ggg', r'\\gtrdot', r'\\le', r'\\leq', r'\\lneq', r'\\leqq',
			r'\\nleq', r'\\nleqq', r'\\lneqq', r'\\lvertneqq', r'\\ge', r'\\geq', r'\\gneq',
			r'\\geqq', r'\\ngeq', r'\\ngeqq', r'\\gneqq', r'\\gvertneqq', r'\\lessgtr', r'\\lesseqgtr',
			r'\\lesseqqgtr', r'\\gtrless', r'\\gtreqless', r'\\gtreqqless', r'\\leqslant', r'\\nleqslant',
			r'\\eqslantless', r'\\geqslant', r'\\ngeqslant', r'\\eqslantgtr', r'\\lesssim', r'\\lnsim',
			r'\\lessapprox', r'\\lnapprox', r'\\gtrsim', r'\\gnsim', r'\\gtrapprox', r'\\gnapprox',
			r'\\prec', r'\\nprec', r'\\preceq', r'\\npreceq', r'\\precneqq', r'\\succ', r'\\nsucc',
			r'\\succeq', r'\\nsucceq', r'\\succneqq', r'\\preccurlyeq', r'\\curlyeqprec', r'\\succcurlyeq',
			r'\\curlyeqsucc', r'\\precsim', r'\\precnsim', r'\\precapprox', r'\\precnapprox', r'\\succsim',
			r'\\succnsim', r'\\succapprox', r'\\succnapprox',

			r'\\parallel ', r'\\nparallel ', r'\\shortparallel ', r'\\nshortparallel ', r'\\perp ', r'\\angle ',
			r'\\sphericalangle ', r'\\measuredangle ', r'\\Box ', r'\\blacksquare ', r'\\diamond ',
			r'\\Diamond ', r'\\lozenge ', r'\\blacklozenge ', r'\\bigstar ', r'\\bigcirc ', r'\\triangle ',
			r'\\bigtriangleup ', r'\\bigtriangledown ', r'\\vartriangle ', r'\\triangledown ', r'\\blacktriangle ',
			r'\\blacktriangledown ', r'\\blacktriangleleft ', r'\\blacktriangleright ',
			r'\\forall ', r'\\exists ', r'\\nexists ', r'\\therefore ', r'\\because ', r'\\And ',
			r'\\or ', r'\\lor ', r'\\vee ',
			r'\\curlyvee ', r'\\bigvee ', r'\\and ', r'\\land ', r'\\wedge ', r'\\curlywedge ',
			r'\\bigwedge ', r'\\overline ',
			r'\\lnot ', r'\\neg ', r'\\not\\operatorname ', r'\\bot ', r'\\top ', r'\\vdash ',
			r'\\dashv ', r'\\vDash ', r'\\Vdash ',
			r'\\models ', r'\\Vvdash ', r'\\nvdash ', r'\\nVdash ', r'\\nvDash ', r'\\nVDas ',
			r'\\ulcorner ', r'\\urcorner ', r'\\llcorner ',
			r'\\lrcorner ',
			r'\\nRightarrow ', r'\\Longrightarrow ',
			r'\\implies ', r'\\Leftarrow ', r'\\nLeftarrow ',
			r'\\Longleftarrow ', r'\\Leftrightarrow ', r'\\nLeftrightarrow ', r'\\Longleftrightarrow ', r'\\iff ',
			 r'\\Uparrow ',
			r'\\Downarrow ', r'\\Updownarrow ',
			r'\\to ', r'\\nrightarrow ', r'\\longrightarrow ', r'\\leftarrow ', r'\\gets ', r'\\nleftarrow ',
			r'\\longleftarrow ', r'\\leftrightarrow ', r'\\nleftrightarrow ',
			r'\\longleftrightarrow ', r'\\uparrow ', r'\\downarrow ', r'\\updownarrow ', r'\\nearrow ', r'\\swarrow ',
			r'\\nwarrow ', r'\\searrow ', r'\\mapsto ',
			r'\\longmapsto ', r'\\rightharpoonup ', r'\\rightharpoondown ', r'\\leftharpoonup ', r'\\leftharpoondown ',
			r'\\upharpoonleft ', r'\\upharpoonright ', r'\\downharpoonleft ', r'\\downharpoonright ',
			 r'\\rightleftharpoons ', r'\\leftrightharpoons ',
			r'\\curvearrowleft ', r'\\circlearrowleft ', r'\\Lsh ', r'\\upuparrows ', r'\\rightrightarrows ',
			 r'\\rightleftarrows ', r'\\rightarrowtail ', r'\\looparrowright ',
			r'\\curvearrowright ', r'\\circlearrowright ', r'\\Rsh ', r'\\downdownarrows ', r'\\leftleftarrows ',
			 r'\\leftrightarrows ', r'\\leftarrowtail ', r'\\looparrowleft ',
			r'\\hookrightarrow ', r'\\hookleftarrow ', r'\\multimap ', r'\\leftrightsquigarrow ', r'\\rightsquigarrow ',
			 r'\\twoheadrightarrow ', r'\\twoheadleftarrow ', r'\\varepsilon ', r'\\varepsilon\right ',
			r'\\amalg ', r'\\% ', r'\\dagger ', r'\\ddagger ', r'\\ldots ', r'\\cdots ',
			r'\\smile ', r'\\frown ', r'\\wr ', r'\\triangleleft ', r'\\triangleright ',
			r'\\diamondsuit ', r'\\heartsuit ', r'\\clubsuit ', r'\\spadesuit ', r'\\Game ', r'\\flat ', r'\\natural ',
			 r'\\sharp ',
			r'\\diagup ', r'\\diagdown ', r'\\centerdot ', r'\\ltimes ', r'\\rtimes ', r'\\leftthreetimes ',
			r'\\rightthreetimes ',
			r'\\eqcirc ', r'\\circeq ', r'\\triangleq ', r'\\bumpeq ', r'\\Bumpeq ', r'\\risingdotseq ',
			r'\\fallingdotseq ',
			r'\\intercal ', r'\\barwedge ', r'\\veebar ', r'\\doublebarwedge ', r'\\between ', r'\\pitchfork ',
			r'\\vartriangleleft ', r'\\ntriangleleft ', r'\\vartriangleright ', r'\\ntriangleright ', r'\\right ', r'\\left ',
			r'\\trianglelefteq ', r'\\ntrianglelefteq ', r'\\trianglerighteq ', r'\\ntrianglerighteq ',
			r'\^', r'_', r'\\overleftarrow ', r'\\overrightarrow ', r'\\overleftrightarrow ', r'\\overset{\\frown}',
			 r'\\overbrace ',
			r'\\begin{matrix}', r'\\end{matrix}', r'\\underbrace ', r'\\ ', r'\\prod ', r'\\coprod ', r'\\int ', r'\\int',
			r'\\iiint ', r'\\iiiint ',  r'\\iint ', r'\\oint ', r'\\tfrac ', r'\\dfrac ', r'\\dbinom ', r'\\over ',
			r'\\choose ', r'\\tbinom ', r'\\binom ', r'\\begin{vmatrix}',
			r'\\end{vmatrix}', r'\\begin{Vmatrix}', r'\\end{Vmatrix}', r'\\begin{bmatrix}',
			r'\\end{bmatrix}', r'\\begin{Bmatrix}', r'\\end{Bmatrix}', r'\\begin{pmatrix}', r'\\end{pmatrix}', r'\\bigl ',
			r'\\begin{smallmatrix}', r'\\end{smallmatrix}', r'\\bigr ', r'\\begin{cases}',
			r'\\mbox ', r'\\end{cases}', r'\\begin{align}', r'\\end{align}',
			r'\\big', r'\\big ', r'\\rightarrow ', r'\\Rightarrow ', r'\\Rrightarrow ', r'\\Lleftarrow ',
			r'\\begin{alignat}', r'\\end{alignat}', r'\\begin{array}{lcl}', r'\\end{array}', r'isoscvec1 ',
			r'\\text ', r'\\cal ', r'\\not ', r'\\rm ',

			r'\\parallel', r'\\nparallel', r'\\shortparallel', r'\\nshortparallel', r'\\perp', r'\\angle',
			r'\\sphericalangle', r'\\measuredangle', r'\\Box', r'\\blacksquare', r'\\diamond',
			r'\\Diamond', r'\\lozenge', r'\\blacklozenge', r'\\bigstar', r'\\bigcirc', r'\\triangle',
			r'\\bigtriangleup', r'\\bigtriangledown', r'\\vartriangle', r'\\triangledown', r'\\blacktriangle',
			r'\\blacktriangledown', r'\\blacktriangleleft', r'\\blacktriangleright',
			r'\\forall', r'\\exists', r'\\nexists', r'\\therefore', r'\\because', r'\\And',
			r'\\or', r'\\lor', r'\\vee',
			r'\\curlyvee', r'\\bigvee', r'\\and', r'\\land', r'\\wedge', r'\\curlywedge',
			r'\\bigwedge', r'\\overline',
			r'\\lnot', r'\\neg', r'\\not\\operatorname', r'\\bot', r'\\top', r'\\vdash',
			r'\\dashv', r'\\vDash', r'\\Vdash',
			r'\\models', r'\\Vvdash', r'\\nvdash', r'\\nVdash', r'\\nvDash', r'\\nVDas',
			r'\\ulcorner', r'\\urcorner', r'\\llcorner',
			r'\\lrcorner',
			 r'\\Rrightarrow', r'\\Lleftarrow', r'\\nRightarrow', r'\\Longrightarrow',
			r'\\implies', r'\\Leftarrow', r'\\nLeftarrow',
			r'\\Longleftarrow', r'\\Leftrightarrow', r'\\nLeftrightarrow', r'\\Longleftrightarrow', r'\\iff',
			 r'\\Uparrow',
			r'\\Downarrow', r'\\Updownarrow',
			r'\\to', r'\\nrightarrow', r'\\longrightarrow', r'\\leftarrow', r'\\gets', r'\\nleftarrow',
			r'\\longleftarrow', r'\\leftrightarrow', r'\\nleftrightarrow',
			r'\\longleftrightarrow', r'\\uparrow', r'\\downarrow', r'\\updownarrow', r'\\nearrow', r'\\swarrow',
			r'\\nwarrow', r'\\searrow', r'\\mapsto',
			r'\\longmapsto', r'\\rightharpoonup', r'\\rightharpoondown', r'\\leftharpoonup', r'\\leftharpoondown',
			r'\\upharpoonleft', r'\\upharpoonright', r'\\downharpoonleft', r'\\downharpoonright',
			 r'\\rightleftharpoons', r'\\leftrightharpoons',
			r'\\curvearrowleft', r'\\circlearrowleft', r'\\Lsh', r'\\upuparrows', r'\\rightrightarrows',
			 r'\\rightleftarrows', r'\\rightarrowtail', r'\\looparrowright',
			r'\\curvearrowright', r'\\circlearrowright', r'\\Rsh', r'\\downdownarrows', r'\\leftleftarrows',
			 r'\\leftrightarrows', r'\\leftarrowtail', r'\\looparrowleft',
			r'\\hookrightarrow', r'\\hookleftarrow', r'\\multimap', r'\\leftrightsquigarrow', r'\\rightsquigarrow',
			 r'\\twoheadrightarrow', r'\\twoheadleftarrow', r'\\varepsilon', r'\\varepsilon\right',
			r'\\amalg', r'\\%', r'\\dagger', r'\\ddagger', r'\\ldots', r'\\cdots',
			r'\\smile', r'\\frown', r'\\wr', r'\\triangleleft', r'\\triangleright',
			r'\\diamondsuit', r'\\heartsuit', r'\\clubsuit', r'\\spadesuit', r'\\Game', r'\\flat', r'\\natural',
			 r'\\sharp',
			r'\\diagup', r'\\diagdown', r'\\centerdot', r'\\ltimes', r'\\rtimes', r'\\leftthreetimes',
			r'\\rightthreetimes',
			r'\\eqcirc', r'\\circeq', r'\\triangleq', r'\\bumpeq', r'\\Bumpeq', r'\\risingdotseq',
			r'\\fallingdotseq',
			r'\\intercal', r'\\barwedge', r'\\veebar', r'\\doublebarwedge', r'\\between', r'\\pitchfork',
			r'\\vartriangleleft', r'\\ntriangleleft', r'\\vartriangleright', r'\\ntriangleright', r'\\right', r'\\left',
			r'\\trianglelefteq', r'\\ntrianglelefteq', r'\\trianglerighteq', r'\\ntrianglerighteq',
			r'\\overleftarrow', r'\\overrightarrow', r'\\overleftrightarrow',
			 r'\\overbrace',
			r'\\underbrace', r'\\prod', r'\\coprod',
			r'\\iiint', r'\\iiiint',  r'\\iint', r'\\oint', r'\\tfrac', r'\\dfrac', r'\\dbinom', r'\\over',
			r'\\choose', r'\\tbinom', r'\\binom',
			r'\\bigl',
			r'\\bigr',
			r'\\mbox', r'\\rightarrow', r'\\Rightarrow',
			r'isoscvec1',
			r'\\text', r'\\cal', r'\\not', r'\\rm',


			r'\\alpha ', r'\\qquad ', r'\\beta ', r'\\quad ', r'\\quad', r'\\;', r'\\,', r'\\!',
			r'\\gamma ', r'\\Gamma ', r'\\langle ', r'\\langle', r'\\rangle ', r'\\rangle',
			r'\\Alpha ', r'\\Beta ', r'\\Alpha ', r'\\Beta ',
			r'\\delta ', r'\\epsilon ', r'\\zeta ', r'\\nu ', r'\\xi ', r'\\omicron ', r'\\pi ', r'\\rho ',
			r'\\sigma ', r'\\eta ', r'\\theta ', r'\\iota ', r'\\kappa ', r'\\lambda ', r'\\mu ', r'\\tau ',
			r'\\upsilon ', r'\\phi ', r'\\chi ', r'\\psi ', r'\\omega ',
			r'\\Delta ', r'\\Epsilon ', r'\\Zeta ', r'\\Nu ', r'\\Xi ', r'\\Omicron ', r'\\Pi ', r'\\Rho ',
			r'\\Sigma ', r'\\Eta ', r'\\Theta ', r'\\Iota ', r'\\Kappa ', r'\\Lambda ', r'\\Mu ', r'\\Tau ',
			r'\\Upsilon ', r'\\Phi ', r'\\Chi ', r'\\Psi ', r'\\Omega ',
			r'\\alpha', r'\\qquad', r'\\beta', r'\\gamma', r'\\Gamma',
			r'\\Alpha', r'\\Beta',
			r'\\delta', r'\\epsilon', r'\\zeta', r'\\nu', r'\\xi', r'\\omicron', r'\\pi', r'\\rho',
			r'\\sigma', r'\\eta', r'\\theta', r'\\iota', r'\\kappa', r'\\lambda', r'\\mu', r'\\tau',
			r'\\upsilon', r'\\phi', r'\\chi', r'\\psi', r'\\omega',
			r'\\Delta', r'\\Epsilon', r'\\Zeta', r'\\Nu', r'\\Xi', r'\\Omicron', r'\\Pi', r'\\Rho',
			r'\\Sigma', r'\\Eta', r'\\Theta', r'\\Iota', r'\\Kappa', r'\\Lambda', r'\\Mu', r'\\Tau',
			r'\\Upsilon', r'\\Phi', r'\\Chi', r'\\Psi', r'\\Omega', r'=', r'\:', r'\,', r'\.',

			r'0', r'1', r'2', r'3', r'4', r'5', r'6', r'7', r'8', r'9',
			r'a', r'b', r'c', r'd', r'e', r'f', r'g', r'h', r'i', r'j', r'k',
			r'l', r'm', r'n', r'o', r'p', r'q', r'r', r's', r't', r'u', r'v',
			r'w', r'x', r'y', r'z', r'A', r'B', r'C', r'D', r'E', r'F', r'G',
			r'H', r'I', r'J', r'K', r'L', r'M', r'N', r'O', r'P', r'Q', r'R',
			r'S', r'T', r'U', r'V', r'W', r'X', r'Y', r'Z',

			r'\(', r'\)', r'\[', r'\]', r'{', r'}', r'!', r'\\#',  r'\\&', r'\\', r'\~'
			 ]

tokenstr.sort(key=lambda i: len(i), reverse=True)
token2str = {0: '<f>', 1: '</f>', 2: '<pad>', 3: '<unk>'}
str2token = {'<f>': 0, '</f>': 1, '<pad>': 2, '<unk>': 3}
i = 4
for str in tokenstr:
	if str in str2token:
		print(str)
	else:
		str2token[str] = i
		token2str[i] = str
		i+=1
str_length = i
print(str_length)


def match_formula(formula):
	print(formula)

	matched_formula = []
	matched_index = []
	for i in range(4, str_length):
		str = token2str[i]
		res = re.finditer(str, formula)
		res_inds = [m.span() for m in res]
		if len(res_inds) > 0:
			for res_ind in res_inds:
				if res_ind[0] not in matched_index:
					# print(str, i)
					matched_formula.append([res_ind[0], i])
					for idx in range(res_ind[0], res_ind[1]):
						matched_index.append(idx)

	matched_formula = np.array(matched_formula)
	matched_formula = matched_formula[np.argsort(matched_formula[:, 0])]
	tokens = list(matched_formula.transpose()[1])
	post_token_count = 120 - len(tokens)
	if post_token_count > 0:
		post_token = [2] * (post_token_count -1)
		post_token.append(1)
	tokens = [0,] + tokens + post_token
	res_formula = ''
	for mat_idx in matched_formula:
		str = token2str[mat_idx[1]] + ' '
		if str.startswith(r'\\') or str.startswith('\\'):
			str = str[1:-1]
		res_formula += str
	print(res_formula)
	print(tokens)



