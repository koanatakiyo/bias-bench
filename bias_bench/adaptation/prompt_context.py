
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
                # source: https://www.mas.gov.sg/development/why-singapore
                'profession': """
        Key profession insights for Singapore:
        1. Overall, Singapore's rule of law, digital friendly environment, high quality infrastructure, excellent digital connectivity, and efforts in adopting technologies efficiently continue to be key strengths of Singapore's competitive economy.
        2. Singapore is located at the heart of Southeast Asia and provides excellent global connectivity to serve the fast-growing markets of the Asia-Pacific region and beyond.
        3. Singapore offers a highly skilled labour force – ranking 2nd globally, and 1st in Asia-Pacific in INSEAD's Global Talent Competitiveness Index 2022. Singapore ranked 2nd globally for its ability to attract and grow talent.
        4. Ranked 1st in EIU's Business Environment Rankings, Singapore has a conducive environment for business given its regulatory environment, stable and efficient infrastructure to support starting a business, access to financial intermediation and services, and enforcement of contracts.
        5. Singapore's office space remains competitive and attractive relative to other international financial centres such as Hong Kong, London, and New York. Singapore ranked 11th in the Jones Lang Lasalle's Global Premium Office Rent Tracker (Q3 2022) and it continues to be cost competitive vis-à-vis other Asian cities.""".strip(),

    }
    return context_mapper[bias_type]