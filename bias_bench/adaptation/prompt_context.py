
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
        7. English Usage by Education Level: Higher education levels correlated with increased English usage at home across all ethnic groups. For university graduates, English was the most spoken language at home for a more than half of the Malays, Chinese, and Indians population.""".strip(),
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
        9. Singles by gender: Singlehood was more common among females university graduates than males. However, singlehood was more common for males than females as a whole.""".strip(),
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
    }
    return context_mapper[bias_type]