use std::collections::{HashMap, HashSet};

pub fn topo_order(mut edges_rev: HashMap<usize, HashSet<usize>>) -> Option<Vec<usize>> {
    let mut edges: HashMap<usize, HashSet<usize>> = edges_rev.keys().map(|idx| (*idx, HashSet::new())).collect();

    for (&idx, parents) in edges_rev.iter() {
        for parent in parents {
            edges.get_mut(parent).unwrap().insert(idx);
        }
    }

    let mut leafs: HashSet<usize> =
        edges_rev.iter().filter_map(|(&idx, parents)| parents.is_empty().then_some(idx)).collect();

    let mut topo = Vec::new();

    loop {
        if leafs.is_empty() {
            break;
        }

        let n = *leafs.iter().next().unwrap();
        leafs.remove(&n);
        topo.push(n);

        let children = edges.get(&n).unwrap().clone();
        for child in children {
            edges.get_mut(&n).unwrap().remove(&child);

            let parents = edges_rev.get_mut(&child).unwrap();
            parents.remove(&n);
            if parents.is_empty() {
                leafs.insert(child);
            }
        }
    }

    (edges.values().all(HashSet::is_empty) && edges_rev.values().all(HashSet::is_empty)).then_some(topo)
}
