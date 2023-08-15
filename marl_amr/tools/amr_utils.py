"""A collection of commonly-used standalone functions relevant to AMR."""

from mfem._ser.gridfunc import ProlongToMaxOrder


def output_mesh(env, output_dir, step, output_order=False):
    """Saves mesh files for later visualization.

    Args:
        env: an AMR env object
        output_dir: str path to target directory
        step: int postfix for file names
    """
    meshfile = '{}/mesh'+str(step)+'.mesh'
    env.solver.mesh.Print(meshfile.format(output_dir), 8)

    if env.solver.name == 'euler':
        for eq in range(env.solver.nvars):
            solfile = '{}/sol{}_{}.gf'.format(output_dir, eq, step)
            sol = ProlongToMaxOrder(env.solver.solution)
            sol.Save(solfile, 8)
    else:
        solfile = '{}/sol'+str(step)+'.gf'
        sol = ProlongToMaxOrder(env.solver.solution)
        sol.Save(solfile.format(output_dir), 8)

    if output_order:
        orderfile = '{}/order'+str(step)+'.gf'
        orders = env.solver.GetOrderGridFunction()
        orders.Save(orderfile.format(output_dir), 8)
