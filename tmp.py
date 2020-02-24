import sys
from conll import reader
from conll import util
from pprint import pprint


lines = ['bc/cctv/00/cctv_0001   0   0   It   PRP   (TOP(S(NP*)   -   -   -   '
 'Speaker#1   *   *   (ARG1*)   (3)\n',
 'bc/cctv/00/cctv_0001   0   1   is   VBZ   (VP*   be   03   -   Speaker#1   '
 '*   (V*)   *   -\n',
 'bc/cctv/00/cctv_0001   0   2   composed   VBN   (VP*   compose   01   2   '
 'Speaker#1   *   *   (V*)   -\n',
 'bc/cctv/00/cctv_0001   0   3   of   IN   (PP*   -   -   -   Speaker#1   *   '
 '*   (ARG2*   -\n',
 'bc/cctv/00/cctv_0001   0   4   a   DT   (NP(NP*   -   -   -   Speaker#1   '
 '*   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   5   primary   JJ   *   -   -   -   Speaker#1   '
 '*   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   6   stele   NN   *)   -   -   -   Speaker#1   *   '
 '*   *   (4)\n',
 'bc/cctv/00/cctv_0001   0   7   ,   ,   *   -   -   -   Speaker#1   *   *   '
 '*   -\n',
 'bc/cctv/00/cctv_0001   0   8   secondary   JJ   (NP*   -   -   -   '
 'Speaker#1   *   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   9   steles   NNS   *)   -   -   -   Speaker#1   '
 '*   *   *   (5)\n',
 'bc/cctv/00/cctv_0001   0   10   ,   ,   *   -   -   -   Speaker#1   *   *   '
 '*   -\n',
 'bc/cctv/00/cctv_0001   0   11   a   DT   (NP*   -   -   -   Speaker#1   *   '
 '*   *   -\n',
 'bc/cctv/00/cctv_0001   0   12   huge   JJ   *   -   -   -   Speaker#1   *   '
 '*   *   -\n',
 'bc/cctv/00/cctv_0001   0   13   round   NN   *   round   -   -   Speaker#1   '
 '*   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   14   sculpture   NN   (NML(NML*)   sculpture   '
 '-   -   Speaker#1   *   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   15   and   CC   *   -   -   -   Speaker#1   *   '
 '*   *   -\n',
 'bc/cctv/00/cctv_0001   0   16   beacon   NN   (NML*   beacon   -   -   '
 'Speaker#1   *   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   17   tower   NN   *)))   -   -   -   Speaker#1   '
 '*   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   18   ,   ,   *   -   -   -   Speaker#1   *   *   '
 '*   -\n',
 'bc/cctv/00/cctv_0001   0   19   and   CC   *   -   -   -   Speaker#1   *   '
 '*   *   -\n',
 'bc/cctv/00/cctv_0001   0   20   the   DT   (NP*   -   -   -   Speaker#1   '
 '(WORK_OF_ART*   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   21   Great   NNP   *   -   -   -   Speaker#1   '
 '*   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   22   Wall   NNP   *)   -   -   -   Speaker#1   '
 '*)   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   23   ,   ,   *   -   -   -   Speaker#1   *   *   '
 '*   -\n',
 'bc/cctv/00/cctv_0001   0   24   among   IN   (PP*   -   -   -   Speaker#1   '
 '*   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   25   other   JJ   (NP*   -   -   -   Speaker#1   '
 '*   *   *   -\n',
 'bc/cctv/00/cctv_0001   0   26   things   NNS   *))))))   -   -   -   '
 'Speaker#1   *   *   *)   -\n',
 'bc/cctv/00/cctv_0001   0   27   .   .   *))   -   -   -   Speaker#1   *   '
 '*   *   -\n']

root= reader.extract_annotated_parse(lines, 0)


line = 'bc/cctv/00/cctv_0001   1   4   Commander   NNP   (NP(NP(NP(NML(NML*)   -   -   -   Speaker#1   *   (ARG0*   -\n'
print(reader.extract_coref_annotation(line))

