//! numpy's legacy global generator (`np.random.seed(int)` + `np.random.permutation`), MT19937,
//! reproduced so that mapclassify's `MaxP` (which reseeds numpy with fixed seeds) matches.

const N: usize = 624;
const M: usize = 397;

/// MT19937 state seeded like `np.random.seed(seed)` for an integer seed (`init_genrand`).
pub struct Mt19937 {
    state: [u32; N],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Mt19937 {
        let mut state = [0u32; N];
        state[0] = seed;
        for i in 1..N {
            let prev = state[i - 1];
            state[i] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(i as u32);
        }
        Mt19937 { state, index: N }
    }

    fn twist(&mut self) {
        for i in 0..N {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % N] & 0x7fff_ffff);
            let mut next = self.state[(i + M) % N] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= 0x9908_b0df;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    /// The next 32-bit output.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= N {
            self.twist();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// numpy's legacy `random_interval(max)`: uniform in `0..=max` by masked rejection.
    fn interval(&mut self, max: u32) -> u32 {
        if max == 0 {
            return 0;
        }
        let mask = u32::MAX >> max.leading_zeros();
        loop {
            let v = self.next_u32() & mask;
            if v <= max {
                return v;
            }
        }
    }

    /// `np.random.permutation(range(n))` after `np.random.seed(seed)`: the legacy Fisher-Yates
    /// shuffle from the end.
    pub fn permutation(&mut self, n: usize) -> Vec<usize> {
        let mut v: Vec<usize> = (0..n).collect();
        for i in (1..n).rev() {
            let j = self.interval(i as u32) as usize;
            v.swap(i, j);
        }
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_outputs_match_numpy_seed_zero() {
        let mut mt = Mt19937::new(0);
        assert_eq!(
            [mt.next_u32(), mt.next_u32(), mt.next_u32()],
            [2_357_136_044, 2_546_248_239, 3_071_714_933]
        );
    }

    #[test]
    fn permutations_match_numpy_legacy() {
        // np.random.seed(s); np.random.permutation(list(range(k))), numpy 2.5.
        let cases: [(u32, &[usize]); 6] = [
            (0, &[2, 0, 1, 3, 4]),
            (0, &[6, 2, 1, 7, 3, 0, 5, 4]),
            (1, &[2, 1, 4, 0, 3]),
            (1, &[7, 2, 1, 6, 0, 4, 3, 5]),
            (7, &[0, 3, 2, 1, 4]),
            (7, &[2, 5, 0, 6, 3, 1, 4, 7]),
        ];
        for (seed, expected) in cases {
            let got = Mt19937::new(seed).permutation(expected.len());
            assert_eq!(got, expected, "seed {seed}");
        }
        assert_eq!(Mt19937::new(0).permutation(3), vec![2, 1, 0]);
        assert_eq!(Mt19937::new(1).permutation(2), vec![0, 1]);
    }
}
