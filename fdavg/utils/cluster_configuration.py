import os
import re
import tensorflow as tf
import argparse
import ipaddress
import socket
import json


# The next 3 functions (i.e., expand_hostlist, _get_slurm_var and _resolve_hostlist)
# are taken from TF repository and changed a bit in order to include the suffix '-ib'
# in SLURM node names used in ARIS
def expand_hostlist(hostlist):

    def split_hostlist(_hostlist):
        """Split hostlist at commas outside of range expressions ('[3-5]')."""
        in_brackets = False
        cur_host = ''
        for c in _hostlist:
            if in_brackets:
                assert c != '['
                if c == ']':
                    in_brackets = False
            elif c == '[':
                in_brackets = True
            elif c == ',':
                assert cur_host != ''
                yield cur_host
                cur_host = ''
                continue
            cur_host += c
        if cur_host:
            yield cur_host

    def expand_range_expression(_range_exp):
        """Expand a range expression like '3-5' to values 3,4,5."""
        for _part in _range_exp.split(','):
            sub_range = _part.split('-')
            if len(sub_range) == 1:
                sub_range = sub_range * 2
            else:
                assert len(sub_range) == 2
            num_digits = len(sub_range[0])
            for i in range(int(sub_range[0]), int(sub_range[1]) + 1):
                yield str(i).zfill(num_digits) + '-ib'

    hosts = []
    try:
        for part in split_hostlist(hostlist):
            # Match prefix (anything but a range expression) and range expression
            # Both are optional
            m = re.match(r'([^,[\]]*)(\[([^\]]+)\])?$', part)
            if m is None:
                raise ValueError('Invalid part: %s' % part)
            prefix = m.group(1) or ''
            if m.group(3) is None:
                hosts.append(prefix + '-ib')
            else:
                hosts.extend(prefix + i for i in expand_range_expression(m.group(3)))
    except Exception as e:
        raise ValueError('Invalid hostlist format "%s": %s' % (hostlist, e))
    return hosts


def _get_slurm_var(name):
    """Gets the SLURM variable from the environment.

    Args:
      name: Name of the step variable

    Returns:
      SLURM_<name> from os.environ
    Raises:
      RuntimeError if variable is not found
    """
    name = 'SLURM_' + name
    try:
        return os.environ[name]
    except KeyError:
        raise RuntimeError('%s not found in environment. '
                           'Not running inside a SLURM step?' % name)


def resolve_hostlist(self):
    """Returns a list of hostnames for nodes running the current job step."""
    return expand_hostlist(_get_slurm_var('STEP_NODELIST'))


def config_slurm_cluster():
    # Create a TensorFlow cluster resolver. ClusterResolvers are a way of specifying cluster
    # information for distributed execution. This is the implementation of ClusterResolver for
    # Slurm clusters. It retrieves system attributes by Slurm environment variables, resolves
    # allocated computing node names, constructs a cluster returns ClusterResolver object which
    # can be used for distributed TensorFlow
    # resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=5000)
    tf.distribute.cluster_resolver.SlurmClusterResolver._resolve_hostlist = resolve_hostlist
    resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(port_base=5000)

    cluster = resolver.cluster_spec()
    job_name, task_index = resolver.get_task_info()

    return task_index, resolver


def get_ip_address():
    """
    Get the IP address of the current machine.

    Returns:
        The IP address of the machine.
    """
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def get_host_name():
    """
    Get the host name of the current machine.

    Returns:
        The host name of the machine.
    """
    host_name = socket.gethostname()
    return host_name


def validate_ip_port(ip_port):
    parts = ip_port.split(':')
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("Invalid IP address and port combination")

    ip, port = parts
    try:
        ipaddress.ip_address(ip)
        port = int(port)
        if port < 0 or port > 65535:
            raise ValueError()
        return ip, port
    except (ValueError, ipaddress.AddressValueError):
        raise argparse.ArgumentTypeError(f"Invalid IP address or port")


def set_task_index_by_name(list_of_node_ports):
    list_of_nodes = [node_port.split(':')[0] for node_port in list_of_node_ports]

    try:
        pos = list_of_nodes.index(get_host_name())
    except ValueError:
        pos = None
        print(f"{get_host_name()} not in the list of node names")

    return pos


def set_task_index(list_of_ip_ports):
    list_of_ips = [ip_port.split(':')[0] for ip_port in list_of_ip_ports]

    try:
        pos = list_of_ips.index(get_ip_address())
    except ValueError:
        pos = None
        print(f"{get_ip_address()} not in the list of ip addresses")

    return pos


def config_custom_cluster(nodes):
    tf_config = {
        'cluster': {
            'worker': nodes
        },
        'task': {'type': 'worker', 'index': set_task_index_by_name(nodes)}
    }
    task_index = tf_config['task']['index']
    os.environ['TF_CONFIG'] = json.dumps(tf_config)

    return task_index, None


def configure_cluster(exper_info):
    if exper_info['slurm']:
        task_index, resolver = config_slurm_cluster()
    else:
        task_index, resolver = config_custom_cluster(exper_info['workers_ip_port'])
    exper_info['task_index'] = task_index
    exper_info['slurm_cluster'] = resolver
