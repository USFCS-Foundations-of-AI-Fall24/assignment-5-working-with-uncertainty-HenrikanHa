## Thanh Cong Nguyen

import random
import argparse
import codecs
import os
import numpy

# Sequence - represents a sequence of hidden states and corresponding
# output variables.

class Sequence:
    def __init__(self, stateseq, outputseq):
        self.stateseq  = stateseq   # sequence of states
        self.outputseq = outputseq  # sequence of outputs
    def __str__(self):
        return ' '.join(self.stateseq)+'\n'+' '.join(self.outputseq)+'\n'
    def __repr__(self):
        return self.__str__()
    def __len__(self):
        return len(self.outputseq)

# HMM model
class HMM:
    def __init__(self, transitions={}, emissions={}):
        """creates a model from transition and emission probabilities
        e.g. {'happy': {'silent': '0.2', 'meow': '0.3', 'purr': '0.5'},
              'grumpy': {'silent': '0.5', 'meow': '0.4', 'purr': '0.1'},
              'hungry': {'silent': '0.2', 'meow': '0.6', 'purr': '0.2'}}"""



        self.transitions = transitions
        self.emissions = emissions

    ## part 1 - you do this.
    def load(self, basename):
        """reads HMM structure from transition (basename.trans),
        and emission (basename.emit) files,
        as well as the probabilities."""
        self.transitions = {}
        with open(f"{basename}.trans", 'r') as trans_file:
            for line in trans_file:
                if not line.strip():
                    continue
                parts = line.strip().split()
                state_from = parts[0]
                state_to = parts[1]
                probability = float(parts[2])
                if state_from not in self.transitions:
                    self.transitions[state_from] = {}
                self.transitions[state_from][state_to] = probability

        self.emissions = {}
        with open(f"{basename}.emit", 'r') as emit_file:
            for line in emit_file:
                if not line.strip():
                    continue
                parts = line.strip().split()
                state = parts[0]
                observation = parts[1]
                probability = float(parts[2])
                if state not in self.emissions:
                    self.emissions[state] = {}
                self.emissions[state][observation] = probability


   ## you do this.
    def generate(self, n):
        """return an n-length Sequence by randomly sampling from this HMM."""
        state_seq = []
        output_seq = []
        current_state = '#'
        for _ in range(n):
            next_states, trans_probs = zip(*self.transitions[current_state].items())
            current_state = random.choices(next_states, weights=trans_probs)[0]
            state_seq.append(current_state)
            outputs, emit_probs = zip(*self.emissions[current_state].items())
            output = random.choices(outputs, weights=emit_probs)[0]
            output_seq.append(output)

        return Sequence(state_seq, output_seq)

    def forward(self, sequence, domain=""):
        state_values = list(self.transitions.keys())
        num_states = len(state_values)
        num_observations = len(sequence)
        M = numpy.zeros((num_states, num_observations + 1))
        state_index = {}
        for i in range(len(state_values)):
            state_index[state_values[i]] = i

        M[state_index['#'], 0] = 1.0
        for state in state_values:
            if state != '#':
                M[state_index[state], 0] = 0.0
                T_start = self.transitions['#'].get(state, 0.0)
                E_emit = self.emissions[state].get(sequence[0], 0.0)
                M[state_index[state], 1] = T_start * E_emit

        for t in range(2, num_observations + 1):
            for state in state_values:
                if state != '#':
                    sum_prob = 0.0
                    E_emit_prev = self.emissions[state].get(sequence[t - 1], 0.0)
                    for state2 in state_values:
                        if state2 != '#':
                            prev_prob = M[state_index[state2], t - 1]
                            T_trans = self.transitions[state2].get(state, 0.0)
                            sum_prob += prev_prob * T_trans * E_emit_prev
                    M[state_index[state], t] = sum_prob

        final_probs = M[:, -1]
        most_likely_state_index = numpy.argmax(final_probs)
        most_likely_state = state_values[most_likely_state_index]
        if domain == "lander":
            safe_landings = {"2,5", "3,4", "4,3", "4,4", "5,5"}
            is_safe = "Yes" if most_likely_state in safe_landings else "No"
            return most_likely_state, is_safe
        return most_likely_state
        ## you do this: Implement the Viterbi algorithm. Given a Sequence with a list of emissions,
        ## determine the most likely sequence of states.

    def viterbi(self, sequence, domain=""):
        state_values = list(self.transitions.keys())
        num_states = len(state_values)
        num_observations = len(sequence)
        M = numpy.zeros((num_states, num_observations + 1))
        Backpointers = numpy.zeros((num_states, num_observations + 1), dtype=int)
        state_index = {}
        for i in range(len(state_values)):
            state_index[state_values[i]] = i

        M[state_index['#'], 0] = 1.0
        for state in state_values:
            if state != '#':
                M[state_index[state], 0] = 0.0
                T_start = self.transitions['#'].get(state, 0.0)
                E_emit = self.emissions[state].get(sequence[0], 0.0)
                M[state_index[state], 1] = T_start * E_emit

        for t in range(2, num_observations + 1):
            for state in state_values:
                if state != '#':
                    E_emit = self.emissions[state].get(sequence[t - 1], 0.0)
                    max_val = float('-inf')
                    best_prev_state = 0

                    for state2 in state_values:
                        if state2 != '#':
                            M_prev = M[state_index[state2], t - 1]
                            T_trans = self.transitions[state2].get(state, 0.0)
                            val = M_prev * T_trans * E_emit
                            if val > max_val:
                                max_val = val
                                best_prev_state = state_index[state2]
                    M[state_index[state], t] = max_val
                    Backpointers[state_index[state], t] = best_prev_state
        final_probs = M[:, -1]
        best_final_state_index = numpy.argmax(final_probs)
        best_final_state = state_values[best_final_state_index]
        most_likely_sequence = [best_final_state]
        for t in range(num_observations, 1, -1):
            best_final_state_index = Backpointers[best_final_state_index, t]
            most_likely_sequence.insert(0, state_values[best_final_state_index])

        if domain == "lander":
            safe_landings = {"2,5", "3,4", "4,3", "4,4", "5,5"}
            is_safe = "Yes" if most_likely_sequence[-1] in safe_landings else "No"
            return most_likely_sequence, is_safe

        return most_likely_sequence
    ## You do this. Given a sequence with a list of emissions, fill in the most likely
    ## hidden states using the Viterbi algorithm.

def main():
    parser = argparse.ArgumentParser(description="Hidden Markov Model Monte Carlo Simulation")
    parser.add_argument("basename", type=str, help="Base name for the .trans and .emit files")
    parser.add_argument("--generate", type=int, help="Generate a random sequence of the specified length")
    parser.add_argument("--forward", type=str, help="Run the forward algorithm on a sequence file")
    parser.add_argument("--viterbi", type=str, help="Run the Viterbi algorithm on a sequence file")

    args = parser.parse_args()

    h = HMM()
    h.load(args.basename)

    if args.generate:
        sequence = h.generate(args.generate)
        print(sequence)
        with open(f"{args.basename}_sequence.obs", "w") as f:
            f.write("\n" + ' '.join(sequence.outputseq) + "\n")
        print(f"Generated output sequence saved to {args.basename}_sequence.obs")

    if args.forward:
        with open(args.forward, 'r') as f:
            sequence = f.read().strip().split()
        print("Observation sequence:", ' '.join(sequence) + "\n")
        result = h.forward(sequence, domain=args.basename)
        if args.basename == "lander":
            most_likely_state, is_safe = result
            print(f"Most likely final state: {most_likely_state}")
            print(f"Safe to land: {is_safe}")
        else:
            print(f"Most likely final state: {result}")

    if args.viterbi:
        with open(args.viterbi, 'r') as f:
            sequence = f.read().strip().split()
        print("Observation sequence:", ' '.join(sequence) + "\n")
        result = h.viterbi(sequence, domain=args.basename)
        if args.basename == "lander":
            most_likely_sequence, is_safe = result
            print("Most likely sequence of states:", ' '.join(most_likely_sequence))
            print(f"Safe to land: {is_safe}")
        else:
            print("Most likely sequence of states:", ' '.join(result))

if __name__ == "__main__":
    main()



