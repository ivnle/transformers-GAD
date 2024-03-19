
if __name__ == "__main__":
    p_1_1 = 3.175825986545533e-05
    p_1_0 = 2.0020976080559194e-05
    p_2_1_given_1 = 0.17334003746509552
    p_2_0_given_1 = 0.22081588208675385
    p_2_0_given_0 = 0.25583988428115845
    p_3_1_given_11 = 0.2095051109790802
    p_3_0_given_11 = 0.06769692897796631
    p_3_1_given_10 = 0.1298251450061798
    p_3_0_given_10 = 0.11201044172048569
    p_3_0_given_00 = 0.21392203867435455

    # Calculating P values using the corrected interpretation for time steps
    P_111 = p_1_1 * p_2_1_given_1 * p_3_1_given_11
    P_110 = p_1_1 * p_2_1_given_1 * p_3_0_given_11
    P_101 = p_1_1 * p_2_0_given_1 * p_3_1_given_10
    P_100 = p_1_1 * p_2_0_given_1 * p_3_0_given_10
    P_000 = p_1_0 * p_2_0_given_0 * p_3_0_given_00

    # Calculating w^s values
    w_s_r = P_111 + P_110 + P_101 + P_100 + P_000
    w_s_a = (p_2_1_given_1) * ((p_3_1_given_11) + (p_3_0_given_11)) + (p_2_0_given_1) * ((p_3_1_given_10) + (p_3_0_given_10))
    w_s_b = (p_3_1_given_11) + (p_3_0_given_11)
    w_s_c = (p_3_1_given_10) + (p_3_0_given_10)
    w_s_d = p_2_0_given_0 * p_3_0_given_00
    w_s_e = p_3_0_given_00

    # Building the dictionary
    success_rates = {
        "root": w_s_r,
        "a": w_s_a,
        "b": w_s_b,
        "c": w_s_c,
        "d": w_s_d,
        "e": w_s_e,
    }

    # Display the success rates
    for non_terminal, success_rate in success_rates.items():
        print(f"{non_terminal}: {success_rate}")

    theta_1_1 = w_s_a / w_s_r
    theta_1_0 = w_s_d / w_s_r
    theta_2_1_given_1 = w_s_b / w_s_a
    theta_2_0_given_1 = w_s_c / w_s_a
    theta_2_0_given_0 = w_s_e / w_s_d
    theta_3_1_given_11 = 1 / w_s_b
    theta_3_0_given_11 = 1 / w_s_b
    theta_3_1_given_10 = 1 / w_s_c
    theta_3_0_given_10 = 1 / w_s_c
    theta_3_0_given_00 = 1 / w_s_e

    # Input IDs
    id_1 = 28740
    id_0 = 28734

    # Mapping input IDs to theta values
    input_id_to_theta = {
        (id_1,): theta_1_1,
        (id_0,): theta_1_0,
        (id_1, id_1): theta_2_1_given_1,
        (id_0, id_1): theta_2_0_given_1,
        (id_0, id_0): theta_2_0_given_0,
        (id_1, id_1, id_1): theta_3_1_given_11,
        (id_1, id_1, id_0): theta_3_0_given_11,
        # Note: This assumes a typo in the question; logically, it should map to theta_3_0_given_11
        (id_1, id_0, id_1): theta_3_1_given_10,
        (id_1, id_0, id_0): theta_3_0_given_10,
        (id_0, id_0, id_0): theta_3_0_given_00,
    }

    # Display the mapping
    for input_ids, theta_value in input_id_to_theta.items():
        print(f"{input_ids}: {theta_value}")