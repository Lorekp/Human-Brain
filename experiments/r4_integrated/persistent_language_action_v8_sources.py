"""Literal public reading material; no generation, imports, calls or hidden labels."""

ADMITTED = (
    "Bay17 receiving, 2026-09-18: cold parcels go to 'cedar'; ordinary parcels go to 'maple'.",
    "Ticket desk, 2026-09-18: allow at most '6' tickets; a negative request receives zero tickets.",
    "Bay17 receiving, 2026-09-19: fragile parcels go to 'wicker', overriding the cold rule.",
    "Regional desks: west caps at '4', east at '7'. "
    "Orders are west/9, east/-2, east/8, in that order.",
)

PROSE = (
    "The packing desk received three labels: '  Fern ', 'oak', and '  '. Its clerk copied "
    "each label into a notebook before removing surrounding whitespace. The recorded "
    "results were 'Fern', 'oak', and an empty string. Letters inside a label were left "
    "alone. This matters for a name such as 'Red Pine': the space between its words is "
    "part of the name. The afternoon operator used the same procedure on a new delivery "
    "rather than substituting yesterday's recorded labels. No alphabetic case change "
    "was requested in this cleaning step.",
    "A measurement log held the values -5, 8, 0, 2, and -1 in arrival order. The report "
    "for strictly positive measurements retained 8 followed by 2. Zero did not qualify, "
    "and removing a value did not reorder the other values. The empty log produced an "
    "empty result. A second log held 4, 4, and -3; its report contained both copies of "
    "4 because the rule concerned the sign, not repetition. The technician wrote down "
    "the selection rule next to the examples so that a later report could use new "
    "measurements without copying these particular numbers.",
    "An attendance sheet listed Inez, Jo, Inez, and Bo. A list of first appearances "
    "contained Inez, Jo, and Bo, in that order. Alphabetizing would have moved Bo to "
    "the front and would have answered a different question. A repeated appearance "
    "does not erase the original position. When the sheet was blank, the summary was "
    "also blank. This method was later applied to badge numbers rather than names; "
    "equal values counted as repeats in either case. The sheet itself was kept intact "
    "so the original sequence of arrivals could still be inspected.",
    "A supply clerk may release between zero and five boxes. A request for -4 boxes "
    "therefore becomes zero, a request for 3 stays 3, and a request for 12 becomes 5. "
    "The upper bound is a parameter of this desk, not a universal shipping rule. A "
    "different desk uses a bound of 9. Applying a bound does not round or multiply a "
    "count. The clerk first checks that the requested count is not below zero, then "
    "keeps it below the desk's upper bound. Both endpoints are allowed, including a "
    "desk whose upper bound is itself zero.",
    "A temporary export had the name ledger.tmp. Removing one terminal .tmp left "
    "ledger. The same operation left ledger.txt unchanged, because the matching "
    "letters must be at the end. It also changed .tmp into an empty name. The name "
    "old.tmp.tmp lost just its final suffix and became old.tmp, not old. The operator "
    "kept capitalization unchanged: LEDGER.TMP did not match the lowercase suffix. "
    "These examples describe a single suffix removal. They do not call for removing "
    "all dots, finding a substring in the middle, or repeatedly applying the rule.",
    "The console displayed a sequence of words using a chosen separator. With words "
    "red, gold, and blue and the separator ':', its display was red:gold:blue. A "
    "one-word sequence displayed the word with no separator on either side. An "
    "empty sequence displayed an empty string. Joining did not sort the words or "
    "discard duplicate words. The separator was supplied by the caller, so another "
    "screen could use a slash without changing the underlying word sequence. The "
    "display routine returned text; printing that text was the caller's next action.",
    "An expense sheet recorded 7, -2, and 4. Adding the entries gave 9, because the "
    "negative entry was a refund. The total for an empty sheet was zero. A second "
    "sheet recorded 6 and 6 and totaled 12; repeated entries were not automatically "
    "duplicates to remove. The accounting assistant distinguished adding entries "
    "from counting entries: this second sheet had two entries but a total of twelve. "
    "It was also different from taking the largest entry. The original values and "
    "their descriptions remained available after the total was calculated.",
    "A display arranged names by their length, shortest first. The names 'wolf', "
    "'ox', 'cat', and 'elk' became 'ox', 'cat', 'elk', and 'wolf'. Cat and elk kept "
    "their relative order because they had equal length. Sorting by length did not "
    "change the spelling of a name. A one-character name can come before a longer "
    "name even when it would come later alphabetically. An empty collection stayed "
    "empty. The curator saved the original arrangement too, because another display "
    "might require arrival order rather than a length-based arrangement.",
    "A lamp's setting depended on a Boolean flag named quiet. When quiet was true, "
    "the chosen label was 'dim'; otherwise the label was 'bright'. The operator "
    "recorded both cases instead of guessing from the word 'lamp'. The value chosen "
    "by the condition could be returned to a caller without being printed. Another "
    "device used different labels with the same conditional shape. Its settings did "
    "not replace the lamp's labels. A conditional choice selects one branch for the "
    "current input; it does not concatenate the text from both branches.",
    "A workshop described a repair by the failing public check and the changed "
    "implementation. The driver imported the implementation module, called its "
    "function with several inputs, and raised AssertionError when an output broke "
    "the stated contract. The maintainer changed the implementation, then ran the "
    "same driver again. A missing import was different from a wrong returned value. "
    "Silencing an assertion would not repair the implementation and the driver was "
    "read-only. Passing the displayed checks established those cases, not every "
    "possible unseen input or every later requirement.",
)

PROGRAMS = (
    "# Remove whitespace only at the edges of one string.\n"
    "# The input is not edited in place; a returned string is used by the caller.\n"
    "def tidy(text):\n    return text.strip()\n\n"
    "examples = [('  Birch ', 'Birch'), ('', ''), (' a b ', 'a b')]\n"
    "for before, expected in examples:\n"
    "    actual = tidy(before)\n"
    "    assert actual == expected, (before, actual, expected)\n"
    "# Internal spaces and upper-case letters are not changed by this function.\n"
    "",
    "# Retain every strictly positive item in the original order.\n"
    "def gains(values):\n    return [value for value in values if value > 0]\n\n"
    "cases = [([-7, 5, 0, 2], [5, 2]), ([], []), ([3, 3], [3, 3])]\n"
    "for values, expected in cases:\n"
    "    original = values[:]\n"
    "    actual = gains(values)\n"
    "    assert actual == expected, (values, actual)\n"
    "    assert values == original, 'the caller still owns the input list'\n"
    "",
    "# Keep first appearances. Dictionary iteration retains insertion order.\n"
    "def first_seen(items):\n    return list(dict.fromkeys(items))\n\n"
    "cases = [(['q', 'p', 'q'], ['q', 'p']), ([], []), ([9, 9], [9])]\n"
    "for items, expected in cases:\n"
    "    result = first_seen(items)\n"
    "    assert result == expected, (items, result)\n"
    "# Sorting the keys would change the contract when the first item is larger.\n"
    "",
    "# A parameter supplies the upper limit; zero is the lower limit.\n"
    "def allowance(count, ceiling):\n    return max(0, min(count, ceiling))\n\n"
    "cases = [(-8, 5, 0), (2, 5, 2), (11, 5, 5), (3, 0, 0)]\n"
    "for count, ceiling, expected in cases:\n"
    "    result = allowance(count, ceiling)\n"
    "    assert result == expected, (count, ceiling, result)\n"
    "# This contract assumes a nonnegative ceiling.\n",
    "# Remove at most one matching suffix, only when it terminates the string.\n"
    "def without_suffix(name, suffix):\n    return name.removesuffix(suffix)\n\n"
    "cases = [('note.log', '.log', 'note'), ('log.note', '.log', 'log.note'),\n"
    "         ('x.log.log', '.log', 'x.log'), ('', '.log', '')]\n"
    "for name, suffix, expected in cases:\n"
    "    actual = without_suffix(name, suffix)\n"
    "    assert actual == expected, (name, suffix, actual)\n"
    "",
    "# The separator is data supplied by the caller, not a fixed output format.\n"
    "def display(words, separator):\n    return separator.join(words)\n\n"
    "cases = [(['up', 'down'], ':', 'up:down'), ([], '/', ''), (['solo'], '|', 'solo')]\n"
    "for words, separator, expected in cases:\n"
    "    actual = display(words, separator)\n"
    "    assert actual == expected, (words, separator, actual)\n"
    "# No leading or trailing separator is inserted.\n",
    "# Add signed entries, with zero as the total for an empty collection.\n"
    "def balance(entries):\n    return sum(entries)\n\n"
    "cases = [([10, -3, 2], 9), ([], 0), ([4, 4], 8), ([-2], -2)]\n"
    "for entries, expected in cases:\n"
    "    actual = balance(entries)\n"
    "    assert actual == expected, (entries, actual, expected)\n"
    "# A repeated entry contributes each time. Negative values subtract from\n"
    "# the balance. This routine does not filter entries.\n",
    "# Sort by string length. Equal keys retain their earlier relative order.\n"
    "def short_first(names):\n    return sorted(names, key=len)\n\n"
    "cases = [(['seal', 'ibis', 'yak', 'ox'], ['ox', 'yak', 'seal', 'ibis']),\n"
    "         ([], []), (['b', 'a'], ['b', 'a'])]\n"
    "for names, expected in cases:\n"
    "    result = short_first(names)\n"
    "    assert result == expected, (names, result)\n"
    "# Alphabetical sorting would exchange b and a.\n",
    "# A condition chooses exactly one of the two supplied values.\n"
    "def choose(flag, when_true, when_false):\n"
    "    return when_true if flag else when_false\n\n"
    "cases = [(True, 'open', 'shut', 'open'), (False, 'up', 'down', 'down'),\n"
    "         (True, 8, 4, 8)]\n"
    "for flag, yes, no, expected in cases:\n"
    "    actual = choose(flag, yes, no)\n"
    "    assert actual == expected, (flag, yes, no, actual)\n"
    "# Returning a value does not print it.\n",
    "# Transform each item without filtering or rearranging the collection.\n"
    "def doubled(values):\n    return [2 * value for value in values]\n\n"
    "cases = [([3, -2, 0], [6, -4, 0]), ([], []), ([1, 1], [2, 2])]\n"
    "for values, expected in cases:\n"
    "    before = values[:]\n"
    "    result = doubled(values)\n"
    "    assert result == expected, (values, result)\n"
    "    assert before == values, 'the source list is unchanged'\n"
    "# The output has one item for each input, including zero and negatives.\n",
)
