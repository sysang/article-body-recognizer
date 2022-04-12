import re

from scrapy import Spider, Request
from newspaper import Article

from scraping.spiders.raw_html_spider import RawHtmlSpider
from scraping.items import ScrapingItem


class ArticleTextSpider(RawHtmlSpider):
    is_shallow = True
    language = None #  Remember to set this attribute for inherited class

    def process_text(self, response):
      assert self.language is not None, "Processing language is not set."

      article = Article(url='', language=self.language)
      html_text= self.parse_response(response)
      article.set_html(html_text)
      article.parse()

      return article.text

    def construct_output(self, response):
        scraped = ScrapingItem()

        scraped['source'] = self.process_url(response)
        scraped['text'] = self.process_text(response)
        scraped['original'] = self.process_original(response)
        scraped['image_urls'] = self.process_image(response)

        return scraped

    def process_image(self, response):
      p = re.compile('(https?\:\/\/)(www\.)?((?:[0-9a-z\-\_]+\.)+)(\w+)\/?.*$')
      repl = lambda matchObj: matchObj.groups()[0] + matchObj.groups()[1] + matchObj.groups()[2] +  matchObj.groups()[3]

      def repl(matchObj):
        groups = matchObj.groups()
        _s = ''
        for gr in groups:
          if gr:
            _s += gr

        return _s

      domain_url = p.sub(repl, response.url)
      img_tags = response.css('img')
      image_urls = []

      for tag in img_tags:
        src = tag.attrib.get('src', None)
        if src:
          if p.search(src):
            image_urls.append(src)
          else:
            image_urls.append(domain_url + src)

      return image_urls


class ArticleTextSobieskiinc(ArticleTextSpider):
    name = "article_text_sobieskiinc"
    allow_domains = ['sobieskiinc.com']
    urls = ["https://www.sobieskiinc.com/blog/page/{}".format(i) for i in range(1, 97)]


class ArticleTexthealthyhomesCesNcsu(ArticleTextSpider):
    name = "article_text_healthyhomes"
    allow_domains = ['healthyhomes.ces.ncsu.edu',]
    urls = ["https://healthyhomes.ces.ncsu.edu/page/{}".format(i) for i in range(1, 6)]


class ArticleTextLowes(ArticleTextSpider):
    name = "article_text_lowes"
    allow_domains = ['lowes.ca']
    urls = [
            'https://www.lowes.ca/ideas-how-to/bathroom-renovation-ideas?sort=score%3Adesc',
            'https://www.lowes.ca/ideas-how-to/flooring?sort=score%3Adesc',
        ]

class ArticleTextMymove(ArticleTextSpider):
    name = "article_text_mymove"
    allow_domains = ['mymove.com']
    urls = [
            'https://www.mymove.com/freshome/',
        ]
    is_shallow = False


class ArticleTextProbuilder(ArticleTextSpider):
    name = "article_text_probuilder"
    allow_domains = ['probuilder.com']
    urls = ["https://www.probuilder.com/blogs?page={}".format(i) for i in range(1, 109)]

class ArticleDutchMisc1(ArticleTextSpider):
  name = "article_dutch_misc1"
  allow_domains = [
        'dakwerkengarant.nl',
        'devogel-dakdekker.nl',
        'dakdekkers-vanleeuwen.nl',
        'multifix-dakdekkers.nl',
      ]
  urls = [
        'https://dakwerkengarant.nl/dak-vervangen/',
        'https://dakwerkengarant.nl/dakreparatie/',
        'https://dakwerkengarant.nl/daklekkage/',
        'https://dakwerkengarant.nl/dak-inspectie/',
        'https://dakwerkengarant.nl/schoorsteen/',
        'https://dakwerkengarant.nl/dakgoten/',

        'https://devogel-dakdekker.nl/diensten/',

        'https://www.dakdekkers-vanleeuwen.nl/dakkapel-renovatie/',
        'https://www.dakdekkers-vanleeuwen.nl/dakkapel-renovatie/',
        'https://www.dakdekkers-vanleeuwen.nl/daklekkage/',
        'https://www.dakdekkers-vanleeuwen.nl/dakisolatie/',
        'https://www.dakdekkers-vanleeuwen.nl/dakdekker-spoed/',
        'https://www.dakdekkers-vanleeuwen.nl/nokvorst/',
        'https://www.dakdekkers-vanleeuwen.nl/vogeloverlast-en-vogelwering/',
        'https://www.dakdekkers-vanleeuwen.nl/dakbedekking/',
        'https://www.dakdekkers-vanleeuwen.nl/plat-dak/',
        'https://www.dakdekkers-vanleeuwen.nl/dak-onderhoud/',
        'https://multifix-dakdekkers.nl/',
      ]


class ArticleDutchMisc2(ArticleTextSpider):
  name = "article_dutch_misc2"
  language = 'nl'
  allow_domains = [
        'groothuisbouw.nl',
        'woonwensfabriek.nl',
        'verbouwkosten.com',
        'selekthuis.nl',
      ]
  urls = [
        'https://www.groothuisbouw.nl/blog',
        'https://www.woonwensfabriek.nl/woongereed',
        'https://www.verbouwkosten.com/',

        'https://selekthuis.nl/selekthuis-blog',
        'https://selekthuis.nl/selekthuis-blog/p2',
        'https://selekthuis.nl/selekthuis-blog/p3',
        'https://selekthuis.nl/selekthuis-blog/p4',
        'https://selekthuis.nl/selekthuis-blog/p5',
        'https://selekthuis.nl/selekthuis-blog/p6',
        'https://selekthuis.nl/selekthuis-blog/p7',
        'https://selekthuis.nl/selekthuis-blog/p8',
        'https://selekthuis.nl/selekthuis-blog/p9',
        'https://selekthuis.nl/selekthuis-blog/p10',
        'https://selekthuis.nl/selekthuis-blog/p11',

        'https://bnla.nl/blog',
      ]


class ArticleVerbouwkosten(ArticleTextSpider):
  name = "article_verbouwkosten"
  language = 'nl'
  allow_domains = ['verbouwkosten.com']
  urls = ["https://www.verbouwkosten.com/blog/page/{}/".format(i) for i in range(1, 213)]

  custom_settings = {
        'ROBOTSTXT_OBEY' : False,
    }


class ArticleInrichtingHuis(ArticleTextSpider):
  name = "article_inrichting_huis"
  language = 'nl'
  allow_domains = ['inrichting-huis.com']
  urls = ["https://www.inrichting-huis.com/page/{}/".format(i) for i in range(1, 377)]


class Article100procentwoongeluk(ArticleTextSpider):
  name = "article_100procentwoongeluk"
  language = 'nl'
  allow_domains = [
        '100procentwoongeluk.nl',
      ]
  urls = [
        'https://100procentwoongeluk.nl/category/bohemian-interieur/',
        'https://100procentwoongeluk.nl/category/botanisch-interieur/',
        'https://100procentwoongeluk.nl/category/botanisch-interieur/page/2/',
        'https://100procentwoongeluk.nl/category/botanisch-interieur/page/3/',
        'https://100procentwoongeluk.nl/category/botanisch-interieur/page/4/',
        'https://100procentwoongeluk.nl/category/botanisch-interieur/page/5/',
        'https://100procentwoongeluk.nl/category/botanisch-interieur/page/6/',
        'https://100procentwoongeluk.nl/category/eclectisch-interieur/',
        'https://100procentwoongeluk.nl/category/eclectisch-interieur/page/2/'
        'https://100procentwoongeluk.nl/category/eclectisch-interieur/page/3/'
        'https://100procentwoongeluk.nl/category/industrieel-interieur/',
        'https://100procentwoongeluk.nl/category/industrieel-interieur/page/2/',
        'https://100procentwoongeluk.nl/category/industrieel-interieur/page/3/',
        'https://100procentwoongeluk.nl/category/industrieel-interieur/page/4/',
        'https://100procentwoongeluk.nl/category/industrieel-interieur/page/5/',
        'https://100procentwoongeluk.nl/category/klassiek-interieur/',
        'https://100procentwoongeluk.nl/category/kleurrijk-interieur/',
        'https://100procentwoongeluk.nl/category/kleurrijk-interieur/page/2/',
        'https://100procentwoongeluk.nl/category/kleurrijk-interieur/page/3/',
        'https://100procentwoongeluk.nl/category/kleurrijk-interieur/page/4/',
        'https://100procentwoongeluk.nl/category/landelijk-interieur/',
        'https://100procentwoongeluk.nl/category/modern-design-interieur/',
        'https://100procentwoongeluk.nl/category/modern-design-interieur/page/2/',
        'https://100procentwoongeluk.nl/category/retro-interieur/',
        'https://100procentwoongeluk.nl/category/romantisch-interieur/',
        'https://100procentwoongeluk.nl/category/scandinavisch-interieur/',
        'https://100procentwoongeluk.nl/category/scandinavisch-interieur/page/2/',
        'https://100procentwoongeluk.nl/category/scandinavisch-interieur/page/3/',
        'https://100procentwoongeluk.nl/category/groen-interieur/',
        'https://100procentwoongeluk.nl/category/groen-interieur/page/2/',
        'https://100procentwoongeluk.nl/category/groen-interieur/page/3/',
        'https://100procentwoongeluk.nl/category/diy/',
        'https://100procentwoongeluk.nl/category/diy/page/2/',
        'https://100procentwoongeluk.nl/category/diy/page/3/',
        'https://100procentwoongeluk.nl/category/diy/page/4/',
        'https://100procentwoongeluk.nl/category/diy/page/5/',
        'https://100procentwoongeluk.nl/category/basis-tips-interieur/',
        'https://100procentwoongeluk.nl/category/basis-tips-interieur/page/2/',
        'https://100procentwoongeluk.nl/category/tuininspiratie/',
        'https://100procentwoongeluk.nl/category/tuininspiratie/page/2/',
      ]


class ArticleWonenonline(ArticleTextSpider):
  name = "article_wonenonline"
  language = 'nl'
  allow_domains = ['wonenonline.nl']
  urls = []
  urls = urls + ["https://www.wonenonline.nl/interieur-inrichten/woonkamer?start={}".format(i * 12) for i in range(0, 3)]
  urls = urls + ["https://www.wonenonline.nl/badkamers?start={}".format(i * 12) for i in range(0, 113)]
  urls = urls + ["https://www.wonenonline.nl/badkamers?start={}".format(i * 12) for i in range(0, 113)]
  urls = urls + ["https://www.wonenonline.nl/keukens?start={}".format(i * 12) for i in range(0, 73)]
  urls = urls + ["https://www.wonenonline.nl/slaapkamers?start={}".format(i * 12) for i in range(0, 40)]
  urls = urls + ["https://www.wonenonline.nl/slaapkamers/kinderkamer?start={}".format(i * 12) for i in range(0, 9)]
  urls = urls + ["https://www.wonenonline.nl/interieur-inrichten/werkkamer?start={}".format(i * 12) for i in range(0, 3)]
  urls = urls + ["https://www.wonenonline.nl/interieur-inrichten/woonstijlen".format(i * 12) for i in range(0, 3)]
  urls = urls + ["https://www.wonenonline.nl/woontrends"]
  urls = urls + ["https://www.wonenonline.nl/interieur-inrichten/woonstijlen".format(i * 12) for i in range(0, 3)]
  urls = urls + ["https://www.wonenonline.nl/interieur-inrichten/interieurtips?start={}".format(i * 12) for i in range(0, 17)]
  urls = urls + ["https://www.wonenonline.nl/interieur-inrichten/kerst-interieur-inspiratie?start={}".format(i * 12) for i in range(0, 12)]
  urls = urls + ["https://www.wonenonline.nl/interieur-inrichten/van-de-redactie?start={}".format(i * 12) for i in range(0, 17)]


class ArticleOntoDutchMisc3(ArticleTextSpider):
  # For Semantic Ontology
  name = "article_onto_dutch_misc3"
  language = 'nl'
  allow_domains = [
        'goedhuis.nl',
        'nu.nl',
      ]

  urls = [
        'https://goedhuis.nl/bekijk-alle-woningen/',

        'https://www.nu.nl/economie',
        'https://www.nu.nl/uit',
        'https://www.nu.nl/wonen',
      ]
