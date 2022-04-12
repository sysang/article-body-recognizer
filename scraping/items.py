# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

from scrapy.item import Item, Field


class ScrapingItem(Item):
  source = Field()
  text = Field()
  original = Field()
  image_urls = Field()
