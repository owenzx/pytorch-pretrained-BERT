import hashlib
import re


class HeadFinder:
    """Compute heads of mentions.
    This class provides functions to compute heads of mentions via modified
    version of the rules that can be found in Michael Collins' PhD thesis.
    The following changes were introduced:
        - handle NML as NP,
        - for coordinated phrases, take the coordination token as head,
    Furthermore, this class provides a function for adjusting heads for proper
    names to multi-token phrases via heuristics (see adjust_head_for_nam).
    """
    def __init__(self):
        self.__nonterminals = ["NP", "NML", "VP", "ADJP", "QP", "WHADVP", "S",
                             "ADVP", "WHNP", "SBAR", "SBARQ", "PP", "INTJ",
                             "SQ", "UCP", "X", "FRAG"]


        self.__nonterminal_rules = {
            "VP": (["TO", "VBD", "VBN", "MD", "VBZ", "VB", "VBG", "VBP", "VP",
                     "ADJP", "NN", "NNS", "NP"], False),
            "ADJP": (["NNS", "QP", "NN", "\$", "ADVP", "JJ", "VBN", "VBG", "ADJP",
              "JJR", "NP", "JJS", "DT", "FW", "RBR", "RBS", "SBAR", "RB"],
                     False),
            "QP": (["\$", "NNS", "NN", "IN", "JJ", "RB", "DT", "CD", "NCD",
              "QP", "JJR", "JJS"], False),
            "WHADVP": (["CC", "WRB"], True),
            "S": (["TO", "IN", "VP", "S", "SBAR", "ADJP", "UCP", "NP"], False),
            "SBAR": (["WHNP", "WHPP", "WHADVP", "WHADJP", "IN", "DT", "S", "SQ",
              "SINV", "SBAR", "FRAG"], False),
            "SBARQ": (["SQ", "S", "SINV", "SBARQ", "FRAG"], False),
            "SQ": (["VBZ", "VBD", "VBP", "VB", "MD", "VP", "SQ"], False),
            "ADVP": (["RB", "RBR", "RBS", "FW", "ADVP", "TO", "CD", "JJR", "JJ",
              "IN", "NP", "JJS", "NN"], True),
            "WHNP": (["WDT", "WP", "WP$", "WHADJP", "WHPP", "WHNP"], True),
            "PP": (["IN", "TO", "VBG", "VBN", "RP", "FW"], True),
            "X": (["S", "VP", "ADJP", "JJP", "NP", "SBAR", "PP", "X"], True),
            "FRAG": (["*"], True),
            "INTJ": (["*"], False),
            "UCP": (["*"], True),
        }

    def get_head(self, tree):
        """
        Compute the head of a mention, which is represented by its parse tree.
        Args:
            tree (nltk.ParentedTree): The parse tree of a mention.
        Returns:
            nltk.ParentedTree: The subtree of the input tree which corresponds
            to the head of the mention.
        """
        head = None

        label = tree.tag
        # print(label)

        if len(tree.children) == 1:
            if tree.height() == 3:
                head = tree.children[0]
            elif tree.height() == 2:
                head = tree
            elif tree.height() == 1:
                head = tree
        elif len(tree.children) == 0:
            head = tree
        elif label in ["NP", "NML"]:
            head = self.__get_head_for_np(tree)
        elif label in self.__nonterminals:
            head = self.__get_head_for_nonterminal(tree)
        if head is None:
            head = self.get_head(tree.children[-1])

        return head

    def __get_head_for_np(self, tree):
        if self.__rule_cc(tree) is not None:
            return self.__rule_cc(tree)
        elif self.__collins_rule_nn(tree) is not None:
            return self.__collins_rule_nn(tree)
        elif self.__collins_rule_np(tree) is not None:
            return self.get_head(self.__collins_rule_np(tree))
        elif self.__collins_rule_nml(tree) is not None:
            return self.get_head(self.__collins_rule_nml(tree))
        elif self.__collins_rule_prn(tree) is not None:
            return self.__collins_rule_prn(tree)
        elif self.__collins_rule_cd(tree) is not None:
            return self.__collins_rule_cd(tree)
        elif self.__collins_rule_jj(tree) is not None:
            return self.__collins_rule_jj(tree)
        elif self.__collins_rule_last_word(tree) is not None:
            return self.__collins_rule_last_word(tree)

    def __get_head_for_nonterminal(self, tree):
        label = tree.tag
        values, traverse_reversed = self.__nonterminal_rules[label]
        if traverse_reversed:
            to_traverse = reversed(tree.children)
        else:
            to_traverse = tree.children
        for val in values:
            for child in to_traverse:
                label = child.tag
                if val == "*" or label == val:
                    if label in self.__nonterminals:
                        return self.get_head(child)
                    else:
                        return self.get_head(child)

    def __rule_cc(self, tree):
        if tree.tag == "NP":
            for child in tree.children:
                if child.tag == "CC":
                    return self.get_head(child)


    def __collins_rule_nn(self, tree):
        for i in range(len(tree.children)-1, -1, -1):
            if re.match("NN|NNP|NNPS|JJR", tree.children[i].tag):
                return self.get_head(tree.children[i])
            elif tree.children[i].tag == "NX":
                return self.get_head(tree.children[i])

    def __collins_rule_np(self, tree):
        for child in tree.children:
            if child.tag == "NP":
                return self.get_head(child)

    def __collins_rule_nml(self, tree):
        for child in tree.children:
            if child.tag == "NML":
                return self.get_head(child)

    def __collins_rule_prn(self, tree):
        for child in tree.children:
            if child.tag == "PRN":
                return self.get_head(child.children[0])

    def __collins_rule_cd(self, tree):
        for i in range(len(tree.children)-1, -1, -1):
            if re.match("CD", tree.children[i].tag):
                return self.get_head(tree.children[i])

    def __collins_rule_jj(self, tree):
        for i in range(len(tree.children)-1, -1, -1):
            if re.match("JJ|JJS|RB", tree.children[i].tag):
                return self.get_head(tree.children[i])
            elif tree.children[i].tag == "QP":
                return self.get_head(tree.children[i])

    def __collins_rule_last_word(self, tree):
        #current_tree = tree.children[-1]
        current_tree = tree
        while current_tree.height() >= 2:
            current_tree = current_tree.children[-1]
        return self.get_head(current_tree)


hf = HeadFinder()


class Mention:
    def __init__(self, doc_name, sent_num, start, end, words):
        self.doc_name = doc_name
        self.sent_num = sent_num
        self.start = start
        self.end = end
        self.words = words 
        self.gold_parse_is_set = False
        self.gold_parse = None
        self.min_spans = set()
        self.head = None
        
        
    def __eq__(self, other):
        if isinstance(other, self.__class__):
            if self.min_spans:
                return self.doc_name == other.doc_name and self.sent_num == other.sent_num \
                       and self.min_spans==other.min_spans
            else:
                return self.doc_name == other.doc_name and self.sent_num == other.sent_num \
                   and self.start == other.start and self.end == other.end 
        return NotImplemented
    
    def __neq__(self, other):
        if isinstance(other, self.__class__):
            return self.__eq__(other)
        
        return NotImplemented
    
    def __str__(self):
        return str("DOC: " +self.doc_name+ ", sentence number: " + str(self.sent_num) 
                   + ", ("+str(self.start)+", " + str(self.end)+")" +
                   (str(self.gold_parse) if self.gold_parse else "") + ' ' + ' '.join(self.words))

    def __hash__(self):
        if self.min_spans:
            return self.sent_num * 1000000 + hash(frozenset(self.min_spans))
        else:
            return self.sent_num * 1000000 + hash(frozenset((self.start, self.end)))

    def get_span(self):
        if self.min_spans:
            ordered_words=[e[0] for e in sorted(self.min_spans, key=lambda e: e[1])]
            return ' '.join(ordered_words)
        else:
            return ' '.join([w[1] for w in self.words])
         
            
    def set_gold_parse(self, tree):
        self.gold_parse = tree
        self.gold_parse_is_set = True

    def are_nested(self, other):
        if isinstance(other, self.__class__):
            if self.__eq__(other):
                return -1
            if True:
                #self is nested in other
                if self.sent_num == other.sent_num and \
                   self.start >= other.start and self.end <= other.end:
                    return 0
                #other is nested in self
                elif self.sent_num == other.sent_num and \
                   other.start >= self.start and other.end <= self.end:
                    return 1
                else:
                    return -1
       
        return NotImplemented


    '''
    This function is for specific cases in which the nodes 
    in the top two level of the mention parse tree do not contain a valid tag.
    E.g., (TOP (S (NP (NP one)(PP of (NP my friends)))))
    '''
    def get_min_span_no_valid_tag(self, root):
        if not root:
            return
        
        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        accepted_tags = None
    
        while queue:
            node, depth = queue.pop(0)

            if not accepted_tags:
                if node.tag[0:2] in ['NP', 'NM']:
                    accepted_tags=['NP', 'NM', 'QP', 'NX']
                elif node.tag[0:2]=='VP':
                    accepted_tags=['VP']

            if node.isTerminal and depth <= terminal_shortest_depth:
                if self.is_a_valid_terminal_node(node.tag, node.pos):
                    self.min_spans.add((node.tag, node.index))
                    terminal_shortest_depth = min(terminal_shortest_depth, depth)
                    
            elif (not self.min_spans or depth < terminal_shortest_depth )and node.children and \
                 (depth== 0 or not accepted_tags or node.tag[0:2] in accepted_tags): 
                for child in node.children:
                    if not child.isTerminal or (accepted_tags and node.tag[0:2] in accepted_tags):
                        queue.append((child, depth+1))    


    """
    Exluding terminals like comma and paranthesis
    """
    def is_a_valid_terminal_node(self, tag, pos):
        if len(tag.split()) == 1:
            if (any(c.isalpha() for c in tag) or \
                any(c.isdigit() for c in tag) or tag == '%') \
                  and (tag != '-LRB-' and tag != '-RRB-') \
                  and pos[0] != 'CC' and pos[0] != 'DT' and pos[0] != 'IN':# not in conjunctions:
                return True
            return False
        else: # for exceptions like ", and"
            for i, tt in enumerate(tag.split()):
                if self.is_a_valid_terminal_node(tt, [pos[i]]):
                    return True
            return False
   

    def get_valid_node_min_span(self, root, valid_tags, min_spans):
        if not root:
            return

        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        while queue:
            node, depth = queue.pop(0)

            if node.isTerminal and depth <= terminal_shortest_depth:
                if self.is_a_valid_terminal_node(node.tag, node.pos):
                    min_spans.add((node.tag, node.index))
                    terminal_shortest_depth = min(terminal_shortest_depth, depth)

            elif (not min_spans or depth < terminal_shortest_depth )and node.children and \
                 (depth== 0 or not valid_tags or node.tag[0:2] in valid_tags):
                for child in node.children:
                    if not child.isTerminal or (valid_tags and node.tag[0:2] in valid_tags):
                        queue.append((child, depth+1))


    def get_top_level_phrases(self, root, valid_tags):
        terminal_shortest_depth = float('inf')
        top_level_valid_phrases = []
        min_spans = set()
        
        if root and root.isTerminal and self.is_a_valid_terminal_node(root.tag, root.pos):
            self.min_spans.add((root.tag, root.index))

        elif root and root.children:
            for node in root.children:
                if node:
                    if node.isTerminal and self.is_a_valid_terminal_node(node.tag, node.pos):
                        self.min_spans.add((node.tag, node.index))
            if not self.min_spans:           
                for node in root.children:
                    if node.children and node.tag[0:2] in valid_tags:
                        top_level_valid_phrases.append(node)

        return top_level_valid_phrases

    def get_valid_tags(self, root):
        valid_tags = None
        NP_tags = ['NP', 'NM', 'QP', 'NX']
        VP_tags = ['VP']

        if root.tag[0:2]=='VP':
            valid_tags = VP_tags
        elif root.tag[0:2] in ['NP', 'NM']:
            valid_tags = NP_tags
        else:
            if root.children: ## If none of the first level nodes are either NP or VP, examines their children for valid mention tags
                all_tags = []
                for node in root.children:
                    all_tags.append(node.tag[0:2])
                if 'NP' in all_tags or 'NM' in all_tags:
                    valid_tags = NP_tags
                elif 'VP' in all_tags:
                    valid_tags = VP_tags
                else:
                    valid_tags = NP_tags

        return valid_tags

    def set_head(self):
        root = self.gold_parse
        assert(root is not None)
        head_node = hf.get_head(root)
        assert(len(head_node.children)<=1)

        if len(head_node.children) == 1:
            head_node = head_node.children[0]
        self.head = [head_node.tag, head_node.index]

    def set_min_span(self):

        if not self.gold_parse_is_set:
            print('The parse tree should be set before extracting minimum spans')
            return NotImplemented

        root = self.gold_parse

        if not root:
            return


        terminal_shortest_depth = float('inf')
        queue = [(root, 0)]

        valid_tags = self.get_valid_tags(root)


        top_level_valid_phrases = self.get_top_level_phrases(root, valid_tags)
        
        if self.min_spans:
            return
        '''
        In structures like conjunctions the minimum span is determined independently
        for each of the top-level NPs
        '''
        # also add head
        try:
            head_node = hf.get_head(root)
        except:
            print(self.start)
            print(self.words)
            exit()
        try:
            assert(len(head_node.children)<=1)
        except:
            print(self.start)
            print(self.words)
            print(head_node.tag)
            print(head_node.children[0].index)
            print([t.tag for t in head_node.children])
            print([t.children for t in head_node.children])
            exit()

        if len(head_node.children) == 1:
            head_node = head_node.children[0]
        self.head = [head_node.tag, head_node.index]

        if top_level_valid_phrases:
            for node in top_level_valid_phrases:
                self.get_valid_node_min_span(node, valid_tags, self.min_spans) 

        else:
            self.get_min_span_no_valid_tag(root)


        """
        If there was no valid minimum span due to parsing errors return the whole span
        """
        if len(self.min_spans)==0:
            self.min_spans.update([(word, self.start + index) for index, word in enumerate(self.words)])


    
class TreeNode:
    def __init__(self, tag, pos, index, isTerminal):
        self.tag = tag
        self.pos = pos
        self.index = index
        self.isTerminal = isTerminal
        self.children = []
        
    def __str__(self, level=0):
        ret = "\t"*level+(self.tag)+"\n"
        for child in self.children:
            ret += child.__str__(level+1)
        return ret

    def get_terminals(self, terminals):
        if self.isTerminal:
            terminals.append(self.tag)
        else:
            for child in self.children:
                child.get_terminals(terminals)

    def refined_get_children(self):    
        children = []
        for child in self.children:
            if not child.isTerminal and child.children and len(child.children)==1 and child.children[0].isTerminal:
                children.append(child.children[0])
            else:
                children.append(child)
        return children

    def height(self):
        max_child_height = 0
        for child in self.children:
            max_child_height = max(max_child_height, child.height())
        return 1 + max_child_height
            

