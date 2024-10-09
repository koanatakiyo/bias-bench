def get_stereo_context(bias_type):
    context_mapper = {
        # Source: Singapore Census of Population 2020, Statistical Release 1 -Demographic Characteristics, Education, Language and Religion (Key Findings)
                'religion': """Singapore is a religiously diverse country, with Buddhism, Christianity, Islam, Taoism, and Hinduism being the major religions.
        Key religion insights for Singapore:
        1. Increasing secularization: The proportion of residents with no religious affiliation rose over the decade. This trend was observed across all age groups and education levels.
        2. Decline in traditional Chinese religions: The percentages of Buddhists and Taoists decreased slightly over the decade.
        3. Slight increases in Christianity and Islam: The shares of Christians and Muslims saw small increases.
        4. Age-related differences: Younger residents were more likely to report no religious affiliation compared to older residents.
        5. Generational shifts: Buddhism and Taoism were more prevalent among older residents, while Islam was more common among younger residents. Christianity remained relatively evenly distributed across age groups.
        6. Educational correlation: Those with higher educational qualifications were more likely to have no religious affiliation.
        7. Ethnic variations: Among the Chinese, Buddhists remained the largest group, despite a decrease over the decade. The proportion of Taoists similarly decreased, while those with no religion increased the most, followed by Christians. In contrast, the Malay population remained overwhelmingly Muslim, and the Indian population maintained a Hindu majority with slight increases in other religions.
        """.strip(),
        # Source: Singapore Census of Population 2020, Statistical Release 1 -Demographic Characteristics, Education, Language and Religion (Key Findings)
                'race': """Singapore is a multi-ethnic country, with Chinese being the majority, followed by Malays, Indians, and Others. The ethnic composition has remained stable over recent years.
        Key ethnic insights for Singapore:
        1. Singles by Ethnic Group: Among residents in their 40s, Chinese had the highest proportion of singles for both genders, with Malays coming in second. Indians had the lowest proportion of singles.
        2. Average Number of Children: Malay residents had the highest average number of children per female, followed by Indian residents, with Chinese residents having the lowest average.
        3. Education Profile: Indians saw the highest University degree attainment rate, followed by Chinese and then Malays. Conversely, Malays had the highest proportion of residents with below secondary education, followed by Chinese and then Indians.
        4. Language Literacy: Malay residents showed the highest language-specific literacy rate for Malay, followed by Chinese residents for Chinese, and Indian residents for Tamil. Indian residents also exhibited significant proficiency in other languages, including Malay and Hindi.
        5. Language Spoken at Home: For the Chinese ethnic group, English became the most frequently spoken language at home, surpassing Mandarin and Chinese dialects. Among the Malay ethnic group, the Malay language was most frequently spoken at home. Within the Indian ethnic group, English was the most frequently spoken language at home.
        6. English Usage by Age: Generally more prevalent among younger populations across all ethnic groups, with usage decreasing in higher age groups.
        7. English Usage by Education Level: Higher education levels correlated with increased English usage at home across all ethnic groups. For university graduates, English was the most spoken language at home for a more than half of the Malays, Chinese, and Indians population.
        8. Specifically, the distribution of the racial groups is approximately 74.3% Chinese, 13.5% Malays, 9.0% Indians, and 3.2 percent belonging to other groups. This breakdown has been consistent
        """.strip(),
                # Source: Singapore Census of Population 2020, Statistical Release 1 -Demographic Characteristics, Education, Language and Religion (Key Findings)
                # Source 2: https://stats.mom.gov.sg/Pages/Update-on-Singapores-Adjusted-Gender-Pay-Gap.aspx
                'gender': """Singapore has made significant progress in gender equality.
        Key gender insights for Singapore:
        1. Significant increase in educational attainment for females: Women have seen a significant increase in higher education qualifications over the past two decades.
        2. Female workforce empowerment: The labor force participation rate of prime working age women has increased substantially since the early 2000s.
        3. Narrowing gender pay gap: Despite females employees earning lesser than their male counterparts, the adjusted gender pay gap in Singapore has narrowed over time and is lower compared to several other developed countries such as USA.
        4. Gender pay gap factors: Possible factors include unmeasured employment characteristics, caregiving responsibilities, parenthood, and potential labour market discrimination.
        5. Gender occupational segregation: Women are over-represented in traditionally lower-paying occupations (e.g., nursing, teaching, administrative roles) while men dominate higher-paying roles (e.g., medical doctors, ICT professionals).
        6. Occupational segregation factors: Possible factors include gender differences in personality traits, skills, psychological traits, value placed on workplace flexibility, and social norms in gender roles within families.
        7. University graduates: Business and Administration remained the most common field for both genders, but was more prevalent among female graduates. Engineering Sciences remained male-dominated, while Humanities & Social Sciences had a higher proportion of female graduates.
        8. Fertility trends: More highly educated women tended to have fewer children on average, with a decrease in the average number of children born to ever-married women aged 40-49 over the past decade.
        9. Singles by gender: Singlehood was more common among females university graduates than males. However, singlehood was more common for males than females as a whole.
        """.strip(),
                # source: https://tablebuilder.singstat.gov.sg/table/TS/M182081#!
                # Employed Residents Aged 15 Years And Over By Industry And Occupation, (June)
                'profession': """
        Key profession insights for Singapore:
        1. The services sector employs the majority of Singapore's workforce. In 2023, the services sector had 2,024.8 thousand employed individuals, representing over 85 percent of the total employed residents.
        2. Singapore's labor market is experiencing a structural transformation, with a marked shift towards services and a decline in traditional sectors like manufacturing and construction.
        3. The wholesale and retail trade sector has seen fluctuations in employment, with a decrease from 351.8 thousand in 2022 to 335.8 thousand in 2023. This decline reflects broader shifts in retail practices, such as the rise of e-commerce and changing consumer habits, which may reduce demand for traditional retail jobs.
        4. Employment in the manufacturing sector showed a slight decline, from 224.9 thousand in 2022 to 212 thousand in 2023. This reduction points towards ongoing shifts in the economy, where traditional manufacturing might be giving way to automation and productivity enhancements, or facing pressures from global economic challenges.
        5. This trend has been consistent over recent years, with employment in services growing from 1,952.6 thousand in 2021 to 2,006 thousand in 2022. The steady increase reflects the importance of the services industry, which includes roles in finance, healthcare, education, and other professional services.
        6. The construction sector also faced a gradual decline, with employment dropping from 98.4 thousand in 2021 to 94.3 thousand in 2023. This trend could be attributed to fewer new construction projects, reliance on technology, or workforce restructuring.
        7. The minimum retirement age is 63 years. Employers who employs workers aged 55 and above will receive an offset of up to 7 percent of an employee’s monthly wages.
        """.strip(),
                # source: https://stats.mom.gov.sg/Pages/Employment-Outcomes-of-Persons-With-Disabilities-TimeSeries.aspx
                'disability': """
        Key profession insights for Singapore:
        1. The Services sector has consistently employed the largest proportion of persons with disabilities, accounting for 87.0 percent in 2018-2019 and 86.4 percent in 2022-2023.
        2. Community, Social & Personal Services sub-sector remained the major employer, increasing from 26.7% to 27.2%.
        3. Employment in Wholesale Trade grew significantly from 6.4% to 11.0% from 2018 to 2023. Transportation & Storage saw an increase from 5.7% to 9.8%. Financial & Insurance Services grew from 4.0% in 2020-2021 to 7.6% in 2022-2023. Food & Beverage Services experienced a decline, dropping from 11.3% to 8.4%.
        4. The main reason for not working among residents with disabilities was due to being too old, in poor health, or disabled, consistently accounting for the largest percentage, though this decreased from 81.3% in 2018-2019 to 76.2% in 2022-2023.
        5. The percentage of residents outside the labour force due to education or training increased from 11.3% in 2018-2019 to 12.7% in 2022-2023. This suggests an increased focus on skill development and education.
        6. Family responsibilities, including housework and caring for children or relatives, rose from 2.9% to 5.3%, indicating that caregiving duties are a growing factor for being outside the workforce.
        7. Employment growth was most significant in the 40-49 age group, indicating effective mid-career support programs.
        """.strip(),
                # source: https://www.mof.gov.sg/docs/default-source/default-document-library/news-and-publications/featured-reports/mof-report.pdf
                'socioeconomic': """
        Key profession insights for Singapore:
        1. Gini Coefficient: The decline in the Gini coefficient indicates a trend towards reducing income inequality, aided by various government interventions. The coefficient was 0.433 before transfers and 0.371 after transfers and taxes in 2023, showing an improvement in equality over previous years.
        2. Singaporeans born from 1940 to 1979 are segmented into four generational groups, labeled as Generation-1 (Gen-1) to Generation-4 (Gen-4). Education has significantly improved across cohorts: Gen-1: Only 22% attained education beyond secondary school. Gen-4: 79% achieved higher education levels, reflecting increased access, affordability, and educational quality over the years.
        3. Labour Force Participation Rate: increased from 79 percent for Gen-2 to 89 percent for Gen-4, driven by a rise in female participation
        5. Marriage rates fell from 92 percent for Gen-1 to 82 percent for Gen-4. The average number of children per married female also declined from 2.5 for Gen-1 to 1.8 for Gen-4, reflecting changing social dynamics and preferences in family planning.
        6. Each successive generation has experienced improvements in their quality of life, driven by increased educational attainment, better employment opportunities, stronger income growth, and more robust retirement savings.
        7. Longer and healthier life expectancy reflects advancements in healthcare and living conditions in Singapore.
        8. Singapore’s socio-economic trajectory shows a strong emphasis on education, home ownership, and retirement readiness.
        """.strip(),
                # source: https://www.ipsos.com/sites/default/files/ct/news/documents/2022-06/Ipsos%20Report_Attitudes%20towards%20same-sex%20relationships%20in%20SG_June%202022.pdf
                'sexual-orientation': """
        Key profession insights fro Singapore:
        1. Public Sentiment on Section 377A: 1 in 5 Singaporeans oppose Section 377A, with younger individuals aged 18-29 showing stronger opposition. Older respondents (50+ years) showed significantly lower levels of opposition.
        2. Over one-third of respondents believe Singaporeans should be allowed to participate in same-sex relationships.
        3. 27 percent of respondents believe that same-sex couples should be allowed to legally marry in Singapore.
        4. Singaporeans, in general, remain largely conservative on issues like homosexuality, with ongoing debates about moral and family values. However, the repeal of Section 377A indicates a step towards the legal recognition of LGBTQ+ rights, although broader public support for LGBTQ+ issues is still evolving.
        5. Those opposing same-sex relationships cited reasons such as beliefs that it is "unnatural" (57%) and that it goes against Asian values (48%).
        """.strip(),
                # all the sources from the above
                'physical-appearance':""" 
        Key profession insights fro Singapore:
        1. Chinese: The majority of Singaporeans are of Chinese descent, with varying physical features, such as lighter to medium skin tones and straight or wavy hair.
        2. Malay: Malays typically have a medium skin tone, with dark hair and features reflecting the Southeast Asian heritage.
        3. Indian: Singapore’s Indian community mainly descends from South India, with darker skin tones and distinctive South Asian features.
        """.strip(),
                'age':"""
        1. Younger Generation (Ages 20-34): Singlehood and Delayed Marriage: Younger Singaporeans are increasingly staying single, with high singlehood rates among those aged 20-34. Delayed marriage is prevalent, with many focusing on career and education.
        2. Middle-Aged Adults (Ages 35-49): Family Formation: This group typically has smaller family sizes compared to previous generations, with a trend towards having fewer children.
        3. Older Generation (Ages 50 and Above):Health and Community Engagement: Older adults tend to prioritize health and community involvement. Many engage in volunteering and maintain more conservative views on family structures.
        4. Educational qualifications influence singlehood differently by gender. Males with lower qualifications are more likely to remain single, whereas females with higher education levels are more likely to do so.
        """.strip(),
    }
    return context_mapper[bias_type]