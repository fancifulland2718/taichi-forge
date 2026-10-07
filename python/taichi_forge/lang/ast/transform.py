from contextvars import ContextVar

from taichi_forge.lang.ast.ast_transformer import ASTTransformer
from taichi_forge.lang.ast.ast_transformer_utils import ASTTransformerContext, UnrollWarning

_unroll_warning = ContextVar("taichi_unroll_warning", default=None)


def transform_tree(tree, ctx: ASTTransformerContext):
    warning = _unroll_warning.get()
    token = None
    if warning is None or ctx.is_real_function:
        warning = UnrollWarning(ctx)
        token = _unroll_warning.set(warning)
    ctx.unroll_warning = warning
    try:
        ASTTransformer()(ctx, tree)
        return ctx.return_data
    finally:
        if token is not None:
            _unroll_warning.reset(token)
