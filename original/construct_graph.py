# This file is adapted from the following sources:
# RepoMap: https://github.com/paul-gauthier/aider/blob/main/aider/repomap.py
# Agentless: https://github.com/OpenAutoCoder/Agentless/blob/main/get_repo_structure/get_repo_structure.py
# grep-ast: https://github.com/paul-gauthier/grep-ast

import colorsys
import os
import random
import sys
import re
import warnings
from collections import Counter, defaultdict, namedtuple
from pathlib import Path
import builtins
import inspect
import networkx as nx
from grep_ast import TreeContext, filename_to_lang
from pygments.lexers import guess_lexer_for_filename
from pygments.token import Token
from pygments.util import ClassNotFound
from tqdm import tqdm
import ast
import pickle
import json
from .utils import create_structure

# tree_sitter is throwing a FutureWarning
warnings.simplefilter("ignore", category=FutureWarning)
from tree_sitter_languages import get_language, get_parser

Tag = namedtuple("Tag", "rel_fname fname line name kind category info".split())


class CodeGraph:

    warned_files = set()

    def __init__(
        self,
        map_tokens=1024,
        root=None,
        main_model=None,
        io=None,
        repo_content_prefix=None,
        verbose=False,
        max_context_window=None,
    ):
        self.io = io
        self.verbose = verbose

        if not root:
            root = os.getcwd()
        self.root = root

        self.max_map_tokens = map_tokens
        self.max_context_window = max_context_window

        # self.token_count = main_model.token_count
        self.repo_content_prefix = repo_content_prefix
        self._structure = None
        self._std_proj_cache = {}
        self._module_callable_cache = {}
        self._builtins_funs = None

    @property
    def structure(self):
        if self._structure is None:
            self._structure = create_structure(self.root)
        return self._structure

    def get_code_graph(self, other_files, mentioned_fnames=None):
        if self.max_map_tokens <= 0:
            return
        if not other_files:
            return
        if not mentioned_fnames:
            mentioned_fnames = set()

        max_map_tokens = self.max_map_tokens

        # With no files in the chat, give a bigger view of the entire repo
        MUL = 16
        padding = 4096
        if max_map_tokens and self.max_context_window:
            target = min(max_map_tokens * MUL, self.max_context_window - padding)
        else:
            target = 0

        tags = self.get_tag_files(other_files, mentioned_fnames)
        code_graph = self.tag_to_graph(tags)

        return tags, code_graph

    def get_tag_files(self, other_files, mentioned_fnames=None):
        try:
            tags = self.get_ranked_tags(other_files, mentioned_fnames)
            return tags
        except RecursionError:
            self.io.tool_error("Disabling code graph, git repo too large?")
            self.max_map_tokens = 0
            return

    def tag_to_graph(self, tags):
        G = nx.MultiDiGraph()

        def _get(tag, attr):
            if isinstance(tag, dict):
                return tag.get(attr)
            return getattr(tag, attr)

        for tag in tags:
            name = _get(tag, 'name')
            if name is None:
                continue
            G.add_node(
                name,
                category=_get(tag, 'category'),
                info=_get(tag, 'info'),
                fname=_get(tag, 'fname'),
                line=_get(tag, 'line'),
                kind=_get(tag, 'kind'),
            )

        for tag in tags:
            if _get(tag, 'category') == 'class':
                class_funcs = (_get(tag, 'info') or '').split('\t')
                for func_name in class_funcs:
                    func_name = func_name.strip()
                    if func_name:
                        G.add_edge(_get(tag, 'name'), func_name)

        tags_ref = [tag for tag in tags if _get(tag, 'kind') == 'ref']
        tags_def = [tag for tag in tags if _get(tag, 'kind') == 'def']
        def_name_counts = Counter(
            _get(tag_def, 'name')
            for tag_def in tags_def
            if _get(tag_def, 'name')
        )

        for tag in tags_ref:
            ref_name = _get(tag, 'name')
            if not ref_name:
                continue

            match_count = def_name_counts.get(ref_name, 0)
            for _ in range(match_count):
                G.add_edge(ref_name, ref_name)
        return G

    def get_rel_fname(self, fname):
        return os.path.relpath(fname, self.root)

    def split_path(self, path):
        path = os.path.relpath(path, self.root)
        return [path + ":"]

    def get_mtime(self, fname):
        try:
            return os.path.getmtime(fname)
        except FileNotFoundError:
            self.io.tool_error(f"File not found error: {fname}")

    def _get_builtin_functions(self):
        if self._builtins_funs is None:
            names = set(dir(builtins))
            names.update(dir(list))
            names.update(dir(dict))
            names.update(dir(set))
            names.update(dir(str))
            names.update(dir(tuple))
            self._builtins_funs = names
        return self._builtins_funs

    def get_class_functions(self, tree, class_name):
        class_functions = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        class_functions.append(item.name)

        return class_functions

    def get_func_block(self, first_line, code_block):
        first_line_escaped = re.escape(first_line)
        pattern = re.compile(rf'({first_line_escaped}.*?)(?=(^\S|\Z))', re.DOTALL | re.MULTILINE)
        match = pattern.search(code_block)

        return match.group(0) if match else None

    def std_proj_funcs(self, code, fname, tree_ast=None):
        """
        write a function to analyze the *import* part of a py file.
        Input: code for fname
        output: [standard functions]
        please note that the project_dependent libraries should have specific project names.
        """
        cache_key = (fname, hash(code))
        cached = self._std_proj_cache.get(cache_key)
        if cached is not None:
            cached_funcs, cached_libs = cached
            return list(cached_funcs), list(cached_libs)

        std_libs = []
        std_funcs = []

        if tree_ast is None:
            try:
                tree = ast.parse(code)
            except Exception:
                self._std_proj_cache[cache_key] = (tuple(), tuple())
                return [], []
        else:
            tree = tree_ast

        codelines = code.split('\n')

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                # identify the import statement
                import_statement = codelines[node.lineno-1]
                for alias in node.names:
                    import_name = alias.name.split('.')[0]
                    if import_name in fname:
                        continue
                    else:
                        # execute the import statement to find callable functions
                        import_statement = import_statement.strip()
                        try:
                            exec(import_statement)
                        except Exception:
                            continue
                        std_libs.append(alias.name)
                        eval_name = alias.name if alias.asname is None else alias.asname
                        try:
                            module_obj = eval(eval_name)
                        except Exception:
                            continue
                        funcs = self._module_callable_cache.get(module_obj)
                        if funcs is None:
                            funcs = [name for name, member in inspect.getmembers(module_obj) if callable(member)]
                            self._module_callable_cache[module_obj] = funcs
                        std_funcs.extend(funcs)

            if isinstance(node, ast.ImportFrom):
                # execute the import statement
                import_statement = codelines[node.lineno-1]
                if node.module is None:
                    continue
                module_name = node.module.split('.')[0]
                if module_name in fname:
                    continue
                else:
                    # handle imports with parentheses
                    if "(" in import_statement:
                        for ln in range(node.lineno-1, len(codelines)):
                            if ")" in codelines[ln]:
                                code_num = ln
                                break
                        import_statement = '\n'.join(codelines[node.lineno-1:code_num+1])
                    import_statement = import_statement.strip()
                    try:
                        exec(import_statement)
                    except Exception:
                        continue
                    for alias in node.names:
                        std_libs.append(alias.name)
                        eval_name = alias.name if alias.asname is None else alias.asname
                        if eval_name == "*":
                            continue
                        try:
                            module_obj = eval(eval_name)
                        except Exception:
                            continue
                        funcs = self._module_callable_cache.get(module_obj)
                        if funcs is None:
                            funcs = [name for name, member in inspect.getmembers(module_obj) if callable(member)]
                            self._module_callable_cache[module_obj] = funcs
                        std_funcs.extend(funcs)
        result = (tuple(std_funcs), tuple(std_libs))
        self._std_proj_cache[cache_key] = result
        return list(result[0]), list(result[1])
                    

    def get_tags(self, fname, rel_fname):
        # Check if the file is in the cache and if the modification time has not changed
        file_mtime = self.get_mtime(fname)
        if file_mtime is None:
            return []
        # miss!
        data = list(self.get_tags_raw(fname, rel_fname))
        return data

    def get_tags_raw(self, fname, rel_fname):
        ref_fname_lst = rel_fname.split('/')
        s = self.structure
        try:
            for fname_part in ref_fname_lst:
                s = s[fname_part]
        except (KeyError, TypeError):
            s = {}

        structure_classes = {
            item.get('name'): item for item in s.get('classes', []) if item.get('name')
        }
        structure_functions = {
            item.get('name'): item for item in s.get('functions', []) if item.get('name')
        }
        structure_class_methods = {}
        for cls in s.get('classes', []):
            for item in cls.get('methods', []):
                name = item.get('name')
                if name:
                    structure_class_methods[name] = item
        structure_all_funcs = {**structure_functions, **structure_class_methods}

        lang = filename_to_lang(fname)
        if not lang:
            return
        language = get_language(lang)
        parser = get_parser(lang)

        # Load the tags queries
        try:
            # scm_fname = resources.files(__package__).joinpath(
            #     "/shared/data3/siruo2/SWE-agent/sweagent/environment/queries", f"tree-sitter-{lang}-tags.scm")
            scm_fname = """
            (class_definition
            name: (identifier) @name.definition.class) @definition.class

            (function_definition
            name: (identifier) @name.definition.function) @definition.function

            (call
            function: [
                (identifier) @name.reference.call
                (attribute
                    attribute: (identifier) @name.reference.call)
            ]) @reference.call
            """
        except KeyError:
            return
        query_scm = scm_fname
        # if not query_scm.exists():
        #     return
        # query_scm = query_scm.read_text()

        with open(str(fname), "r", encoding="utf-8") as f:
            code = f.read()
        codelines = code.splitlines()

        # hard-coded edge cases
        code = code.replace('\ufeff', '')
        code = code.replace('constants.False', '_False')
        code = code.replace('constants.True', '_True')
        code = code.replace("False", "_False")
        code = code.replace("True", "_True")
        code = code.replace("DOMAIN\\username", "DOMAIN\\\\username")
        code = code.replace("Error, ", "Error as ")
        code = code.replace('Exception, ', 'Exception as ')
        code = code.replace("print ", "yield ")
        pattern = r'except\s+\(([^,]+)\s+as\s+([^)]+)\):'
        # Replace 'as' with ','
        code = re.sub(pattern, r'except (\1, \2):', code)
        code = code.replace("raise AttributeError as aname", "raise AttributeError")

        # code = self.io.read_text(fname)
        if not code:
            return
        tree = parser.parse(bytes(code, "utf-8"))
        try:
            tree_ast = ast.parse(code)
        except Exception:
            tree_ast = None

        try:
            std_funcs, std_libs = self.std_proj_funcs(code, fname, tree_ast)
        except Exception:
            std_funcs, std_libs = [], []

        builtins_funs = self._get_builtin_functions()

        # Run the tags queries
        query = language.query(query_scm)
        captures = query.captures(tree.root_node)
        captures = list(captures)

        saw = set()
        for node, tag in captures:
            if tag.startswith("name.definition."):
                kind = "def"
            elif tag.startswith("name.reference."):
                kind = "ref"
            else:
                continue

            saw.add(kind)
            cur_cdl = codelines[node.start_point[0]]
            category = 'class' if 'class ' in cur_cdl else 'function'
            tag_name = node.text.decode("utf-8")
            
            #  we only want to consider project-dependent functions
            if tag_name in std_funcs:
                continue
            elif tag_name in std_libs:
                continue
            elif tag_name in builtins_funs:
                continue

            if category == 'class':
                # try:
                #     class_functions = self.get_class_functions(tree_ast, tag_name)
                # except:
                #     class_functions = "None"
                class_entry = structure_classes.get(tag_name)
                if class_entry:
                    class_functions = [item['name'] for item in class_entry.get('methods', [])]
                    if kind == 'def':
                        line_nums = [
                            class_entry.get('start_line', node.start_point[0] + 1),
                            class_entry.get('end_line', node.end_point[0] + 1),
                        ]
                    else:
                        line_nums = [node.start_point[0], node.end_point[0]]
                else:
                    class_functions = self.get_class_functions(tree_ast, tag_name) if tree_ast else []
                    if kind == 'def':
                        line_nums = [node.start_point[0] + 1, node.end_point[0] + 1]
                    else:
                        line_nums = [node.start_point[0], node.end_point[0]]
                result = Tag(
                    rel_fname=rel_fname,
                    fname=fname,
                    name=tag_name,
                    kind=kind,
                    category=category,
                    info='\n'.join(class_functions), # list unhashable, use string instead
                    line=line_nums,
                )

            elif category == 'function':

                if kind == 'def':
                    func_entry = structure_all_funcs.get(tag_name)
                    if func_entry:
                        cur_cdl = '\n'.join(func_entry.get('text', []))
                        line_nums = [
                            func_entry.get('start_line', node.start_point[0] + 1),
                            func_entry.get('end_line', node.end_point[0] + 1),
                        ]
                    else:
                        start_line = node.start_point[0]
                        end_line = node.end_point[0]
                        cur_cdl = ''.join(codelines[start_line:end_line + 1]).strip('\n')
                        line_nums = [start_line + 1, end_line + 1]
                else:
                    line_nums = [node.start_point[0], node.end_point[0]]

                result = Tag(
                    rel_fname=rel_fname,
                    fname=fname,
                    name=tag_name,
                    kind=kind,
                    category=category,
                    info=cur_cdl,
                    line=line_nums,
                )

            yield result

        if "ref" in saw:
            return
        if "def" not in saw:
            return

        # We saw defs, without any refs
        # Some tags files only provide defs (cpp, for example)
        # Use pygments to backfill refs

        try:
            lexer = guess_lexer_for_filename(fname, code)
        except ClassNotFound:
            return

        tokens = list(lexer.get_tokens(code))
        tokens = [token[1] for token in tokens if token[0] in Token.Name]

        for token in tokens:
            yield Tag(
                rel_fname=rel_fname,
                fname=fname,
                name=token,
                kind="ref",
                line=-1,
                category='function',
                info='none',
            )

    def get_ranked_tags(self, other_fnames, mentioned_fnames):
        # defines = defaultdict(set)
        # references = defaultdict(list)
        # definitions = defaultdict(set)
        
        tags_of_files = list()

        personalization = dict()

        fnames = set(other_fnames)
        # chat_rel_fnames = set()

        fnames = sorted(fnames)

        # Default personalization for unspecified files is 1/num_nodes
        # https://networkx.org/documentation/stable/_modules/networkx/algorithms/link_analysis/pagerank_alg.html#pagerank
        personalize = 10 / len(fnames)

        for fname in tqdm(fnames):
            if not Path(fname).is_file():
                if fname not in self.warned_files:
                    if Path(fname).exists():
                        self.io.tool_error(
                            f"Code graph can't include {fname}, it is not a normal file"
                        )
                    else:
                        self.io.tool_error(f"Code graph can't include {fname}, it no longer exists")

                self.warned_files.add(fname)
                continue

            # dump(fname)
            rel_fname = self.get_rel_fname(fname)

            # if fname in chat_fnames:
            #     personalization[rel_fname] = personalize
            #     chat_rel_fnames.add(rel_fname)

            if fname in mentioned_fnames:
                personalization[rel_fname] = personalize
            
            tags = list(self.get_tags(fname, rel_fname))

            tags_of_files.extend(tags)

            if tags is None:
                continue

        return tags_of_files
    

    def render_tree(self, abs_fname, rel_fname, lois):
        key = (rel_fname, tuple(sorted(lois)))

        if key in self.tree_cache:
            return self.tree_cache[key]

        # code = self.io.read_text(abs_fname) or ""
        with open(str(abs_fname), "r", encoding='utf-8') as f:
            code = f.read() or ""

        if not code.endswith("\n"):
            code += "\n"

        context = TreeContext(
            rel_fname,
            code,
            color=False,
            line_number=False,
            child_context=False,
            last_line=False,
            margin=0,
            mark_lois=False,
            loi_pad=0,
            # header_max=30,
            show_top_of_file_parent_scope=False,
        )

        context.add_lines_of_interest(lois)
        context.add_context()
        res = context.format()
        self.tree_cache[key] = res
        return res

    def to_tree(self, tags, chat_rel_fnames):
        if not tags:
            return ""

        tags = [tag for tag in tags if tag[0] not in chat_rel_fnames]
        tags = sorted(tags)

        cur_fname = None
        cur_abs_fname = None
        lois = None
        output = ""

        # add a bogus tag at the end so we trip the this_fname != cur_fname...
        dummy_tag = (None,)
        for tag in tags + [dummy_tag]:
            this_rel_fname = tag[0]

            # ... here ... to output the final real entry in the list
            if this_rel_fname != cur_fname:
                if lois is not None:
                    output += "\n"
                    output += cur_fname + ":\n"
                    output += self.render_tree(cur_abs_fname, cur_fname, lois)
                    lois = None
                elif cur_fname:
                    output += "\n" + cur_fname + "\n"
                if type(tag) is Tag:
                    lois = []
                    cur_abs_fname = tag.fname
                cur_fname = this_rel_fname

            if lois is not None:
                lois.append(tag.line)

        # truncate long lines, in case we get minified js or something else crazy
        output = "\n".join([line[:100] for line in output.splitlines()]) + "\n"

        return output


    def find_src_files(self, directory):
        if not os.path.isdir(directory):
            return [directory]

        src_files = []
        for root, dirs, files in os.walk(directory):
            for file in files:
                src_files.append(os.path.join(root, file))
        return src_files
    

    def find_files(self, dir):
        chat_fnames = []

        for fname in dir:
            if Path(fname).is_dir():
                chat_fnames += self.find_src_files(fname)
            else:
                chat_fnames.append(fname)
        
        chat_fnames_new = []
        for item in chat_fnames:
            # filter out non-python files
            if not item.endswith('.py'):
                continue
            else:
                chat_fnames_new.append(item)
    
        return chat_fnames_new
    

def get_random_color():
    hue = random.random()
    r, g, b = [int(x * 255) for x in colorsys.hsv_to_rgb(hue, 1, 0.75)]
    res = f"#{r:02x}{g:02x}{b:02x}"
    return res


if __name__ == "__main__":

    dir_name = sys.argv[1]
    # dir_name = "./playground/astropy"
    code_graph = CodeGraph(root=dir_name)
    chat_fnames_new = code_graph.find_files([dir_name])

    tags, G = code_graph.get_code_graph(chat_fnames_new)

    print("---------------------------------")
    print(f"🏅 Successfully constructed the code graph for repo directory {dir_name}")
    print(f"   Number of nodes: {len(G.nodes)}")
    print(f"   Number of edges: {len(G.edges)}")
    print("---------------------------------")

    with open(f'{os.getcwd()}/graph.pkl', 'wb') as f:
        pickle.dump(G, f)
    
    for tag in tags:
        with open(f'{os.getcwd()}/tags.json', 'a+') as f:
            line = json.dumps({
                "fname": tag.fname,
                'rel_fname': tag.rel_fname,
                'line': tag.line,
                'name': tag.name,
                'kind': tag.kind,
                'category': tag.category,
                'info': tag.info,
            })
            f.write(line+'\n')
    print(f"🏅 Successfully cached code graph and node tags in directory ''{os.getcwd()}''")
